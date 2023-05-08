import os
from typing import List
from PIL import Image
from lpw_stable_diffusion import get_weighted_text_embeddings

import torch
from cog import BasePredictor, Input, Path
from diffusers.utils import load_image
from diffusers import (
    StableDiffusionControlNetPipeline,
    EulerAncestralDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
    ControlNetModel
)

# MODEL_ID refers to a diffusers-compatible model on HuggingFace
# e.g. prompthero/openjourney-v2, wavymulder/Analog-Diffusion, etc
MODEL_ID = "readyplayerme/rpm_characters_concepts"
MODEL_CONTROL_NET_ID = "lllyasviel/control_v11p_sd15_openpose"
MODEL_CACHE = "diffusers-cache"
final_size = 1024
nsfw_police_officer = Path("assets/nsfw_police_officer.jpg")


def generate_image(pipe_text2image, pipe_img2img, pose, prompt,
                   negative_prompt, face_prompt, face_negative_prompt, mask, seed):
    prompt_embeds, negative_prompt_embeds = get_weighted_text_embeddings(
        pipe_text2image, prompt, negative_prompt)
    generator = torch.Generator("cuda").manual_seed(seed)
    output = pipe_text2image(num_inference_steps=20,
                             image=pose,
                             generator=generator,
                             prompt_embeds=prompt_embeds,
                             negative_prompt_embeds=negative_prompt_embeds,
                             guidance_scale=5, controlnet_conditioning_scale=1.5)
    if output.nsfw_content_detected:
        raise Exception(
            f"NSFW content detected. Try running it again, or try a different prompt.")

    image = output.images[0]
    prompt_embeds, negative_prompt_embeds = get_weighted_text_embeddings(
        pipe_img2img, prompt, negative_prompt)

    image_upscaled = image.resize((final_size, final_size))
    image_upscaled = pipe_img2img(image=image_upscaled, generator=generator, prompt_embeds=prompt_embeds,
                                  negative_prompt_embeds=negative_prompt_embeds,
                                  strength=0.26, guidance_scale=7, num_inference_steps=20).images[0]

    face = image_upscaled.crop((80, 0, 80 + 250, 0 + 250))
    prompt_embeds, negative_face_prompt = get_weighted_text_embeddings(
        pipe=pipe_img2img, prompt=face_prompt, uncond_prompt=face_negative_prompt)
    face = face.resize((512, 512))
    face_upscaled = pipe_img2img(image=face, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_face_prompt,
                                 strength=0.26, guidance_scale=4, num_inference_steps=30).images[0]
    blended_image = blend_images(image_upscaled, face_upscaled, mask, 80, 0)
    return blended_image


def blend_images(base_image, overlay_image, mask, x, y):
    overlay_image = overlay_image.resize((250, 250))
    blended_overlay = Image.composite(overlay_image,
                                      base_image.crop((x, y, x + overlay_image.width, y + overlay_image.height)), mask)

    blended_image = base_image.copy()
    blended_image.paste(blended_overlay, (x, y))

    return blended_image


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.controlnet = ControlNetModel.from_pretrained(
            MODEL_CONTROL_NET_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=False, torch_dtype=torch.float16
        )
        self.pipe_controlnet = StableDiffusionControlNetPipeline.from_pretrained(MODEL_ID,
                                                                                 cache_dir=MODEL_CACHE,
                                                                                 local_files_only=True,
                                                                                 controlnet=self.controlnet,
                                                                                 torch_dtype=torch.float16).to("cuda")

        self.pipe_controlnet.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe_controlnet.scheduler.config)
        self.pipe_controlnet.enable_model_cpu_offload()
        self.pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID,
                                                                           cache_dir=MODEL_CACHE,
                                                                           safety_checker=None,
                                                                           requires_safety_checker=False,
                                                                           local_files_only=True,
                                                                           torch_dtype=torch.float16).to("cuda")

        self.pipe_img2img.enable_model_cpu_offload()
        self.pipe_img2img.safety_checker = None
        self.female_pose = load_image("assets/female.png")
        self.male_pose = load_image("assets/male.png")
        self.mask = Image.open("assets/mask_250.png").convert("RGBA")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="japanese clothes, Jojo style, Curly top haircut",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="handbag",
        ),
        body_type: str = Input(
            description="Choose body type. Only 2 are available at the moment.",
            choices=["feminine", "masculine"], default="feminine"),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if body_type == "feminine":
            pose = self.female_pose
            keyword = "femalerpm"
            negative_word = "male, man"
        else:
            pose = self.male_pose
            keyword = "masculine malerpm"
            negative_word = "feminine, female"

        base_prompt = f"cute ((3d render)) of (({prompt})), (({keyword})), (trending on artstation, 4k),  ((crisp lines)), high contrast, 3d cinematic quality light, studio light,"\
            "rim light, trending on artstation, 8k,  smooth, sharp focus, 8 k, octane render, rendered in octane, clean background, #3D #Art #DigitalArt #Sculpture"
        base_negative_prompt = f"(({negative_prompt})), {negative_word}, (((monochrome))), blurry, border, frame, blurry, pixelated, low quality, noisy background, focus on chest, naked chest, blurry, border, frame, blurry, pixelated, low quality, uncovered ass, censored, branded, brand, boring outfit, extra face, noisy background, big head, focused on chest, ugly, bad proportions"
        negative_face_prompt = f"{negative_word}, low-quality, realistic photo, creepy face, naked, tits, naked torso, ugly, smashed face, cartoon, open mouth, borders, several frames, ((ugly)), ((morbid)), (mutilated), extra fingers, mutated hands, (poorly drawn face), (deformed), blurry, (bad anatomy), (bad proportions), (extra limbs), cloned face, out of frame, (malformed limbs), (missing arms), (missing legs), (extra arms), (extra legs),"\
            "mutated hands, (fused fingers), (too many fingers), (long neck)"
        prompt_face = f"detailed cute rpm face, {base_prompt}"
        try:
            output = generate_image(pipe_text2image=self.pipe_controlnet, pipe_img2img=self.pipe_img2img, pose=pose, prompt=base_prompt,
                                    negative_prompt=base_negative_prompt, face_prompt=prompt_face,
                                    face_negative_prompt=negative_face_prompt, mask=self.mask, seed=seed)
        except Exception as e:
            return [nsfw_police_officer]

        output_paths = []
        output_path = f"/tmp/output.png"
        output.save(output_path)
        output_paths.append(Path(output_path))

        return output_paths
