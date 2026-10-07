import torch #Ejecutar operaciones matemáticas del modelo
from diffusers import AutoPipelineForText2Image
# Diffusers es una librería especializada en modelos
# Generativos


print("Cargando el modelo ...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32 # float16 o float 32
    # torch.floatXX, indica la precisión numérica
    # con que se cargan los pesos (decimales, la precisión
    # para generar la imagen)
)

modelo = modelo.to("cpu")

prompt = input("Escribe el prompt de la imagen que quieres crear: ")
negative_prompt = "ugly, deformed, disfigured, poor quality, bad anatomy, low quality, worst quality, lowres, blurry, out of frame, extra limbs, poorly drawn hands, poorly drawn feet, poorly drawn face, mutated hands, bad proportions, cloned face, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, bad eyes, crossed eyes, text, watermark, signature, username, error, logo, words, letters, cropped, cut off, draft, grain, pixelated, jpeg artifacts, low contrast, overexposed, underexposed, distortion, amateur"

print("Creando imagen ...")

imagen = modelo(
    prompt = prompt, # Instrucciones o detalle para crear la imagen
    negative_prompt=negative_prompt,
    num_inference_steps=10, #Las veces que va a tratar de crear o refinar la imagen
    guidance_scale=1.0, #Qué tan fiel debe ser al prompt (creatividad, temperatura)
    height=1024,
    width=1024
).images[0]

imagen.save("imagen.png")

print("Imagen guardada.")
