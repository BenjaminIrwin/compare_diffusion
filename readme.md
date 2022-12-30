# Compare Diffusion

This simple tool is built on top of Hugging Face's Diffusers library and helps you visually compare the results of 
different image diffusion parameters (and models). It performs inference based on the parameters you provide and then 
displays the results in a grid, outputted as a PDF.

## Usage

Compare Diffusion takes the following arguments:

* `hf_token` (required): Your Hugging Face token.
* `output_path` (required): The path where you would like to save the output PDF.
* `type` (required): The type of inference you would like to perform. You can choose from `img2img`, `txt2img`, or 
`inpaint`.
* `rows` (required): The dimension you would like to use for the rows of the image matrix. You can choose from `model`,
`image`, `cfg_scale`, `denoising_strength`, `prompt`, `negative_prompt`, `seed`.
* `cols` (required): The dimension you would like to use for the columns of the image matrix. You can choose from the 
same options as for `rows`.
* `model` (required): The selected model(s) you would like to use. You can provide a path to a CKPT file, or a Hugging 
Face model name.
* `cfg_scale` (required): The selected cfg_scale(s) you would like to use.
* `denoising_strength` (optional): The selected denoising_strength(s) you would like to use. Defaults to `[0.0]` if not 
provided.
* `prompt` (required): The selected prompt(s) you would like to use.
* `negative_prompt` (optional): The selected negative_prompt(s) you would like to use. Defaults to `['']` if not provided.
* `seed` (optional): The selected seed(s) you would like to use. Defaults to `[1]` if not provided.
* `height` (optional): The height of the input/output images. Defaults to `512`.
* `width` (optional): The width of the input/output images. Defaults to `512`.
* `inpaint_full_res` (optional): Whether to inpaint at full resolution. Defaults to `False`.
* `inpaint_full_res_padding` (optional): Padding for inpainting at full resolution. Defaults to `35`.

An example of how you might use the Compare Diffusion tool:

`python compare_diffusion.py * hf_token <your_hugging_face_token> * output_path output.pdf * type img2img * rows prompt * cols model * model <model_1> <model_2> * cfg_scale <cfg_scale_1> <cfg_scale_2> * prompt <prompt_1> <prompt_2>`

## Additional Notes
* If you choose `txt2img` as the inference type, the `denoising_strength` argument will be automatically set to [0.0].
* If you choose `img2img` or `inpaint`, you will need to provide input images in the `input/images` directory. Each image
must have the prefix `i_` (for example, `i_1.png`, `i_2.png` etc.).
* If you choose `inpaint`, you will need to provide a mask in the `input/masks` directory. The mask should be a black and
white image where the white pixels represent the area you would like to inpaint. Each mask must have the prefix `m_`
(for example, `m_1.png`, `m_2.png` etc.). The mask should be the same size as the corresponding input image.