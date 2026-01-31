import os
from PIL import Image
from reportlab.pdfgen import canvas

# === PATHS ===
#mapPath = "/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/diffusionPen/generatedData2/"

#basePath = "/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess/"
#mapPath = "/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/diffusionPen/withGrad_400/"

#basePath = "/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/diffusionPen/conditional//"
#mapPath = "/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/diffusionPen/unconditional//"

basePath = "/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/diffusionPen/eta_10//"
mapPath = "/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/diffusionPen/eta_11//"


pdfPath = "./pdf/"
os.makedirs(pdfPath, exist_ok=True)

# === SETTINGS ===
target_height = 100
spacing = 40
margin = 30
page_width = int(800 * 1.2)  # Increased by 20%
page_height = target_height + 100

# === PREPARE OUTPUT ===
#output_pdf_path = os.path.join(pdfPath, "proximal_images1.pdf")
#output_pdf_path = os.path.join(pdfPath, "condition_unconditinal.pdf")
output_pdf_path = os.path.join(pdfPath, "clamp.pdf")


c = canvas.Canvas(output_pdf_path, pagesize=(page_width, page_height))
c.setFont("Helvetica", 8)

# === LOAD IMAGES ===
all_images = sorted([
    f for f in os.listdir(mapPath)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

#print("\n\t all_images:",all_images)
for indx,image_name in enumerate(all_images):
    
    if indx>1000:
        break
    
    gen_img_path = os.path.join(mapPath, image_name)
    base_name = gen_img_path.split("/")[-1]#image_name.split("_")[0] + ".png"
    base_img_path = os.path.join(basePath, base_name)

    try:
        # Load generated image
        gen_img = Image.open(gen_img_path).convert("RGB")
        scale = target_height / gen_img.height
        gen_width = int(gen_img.width * scale)
        gen_img = gen_img.resize((gen_width, target_height))

        # Load original image (if exists)
        if not os.path.exists(base_img_path):
            print(f"Missing base image: {base_img_path}")
            continue
        base_img = Image.open(base_img_path).convert("RGB")
        scale = target_height / base_img.height
        base_width = int(base_img.width * scale)
        base_img = base_img.resize((base_width, target_height))

        # Save temp images
        gen_temp = os.path.join(pdfPath, f"temp_gen_{image_name}.jpg")
        base_temp = os.path.join(pdfPath, f"temp_base_{base_name}.jpg")
        gen_img.save(gen_temp)
        base_img.save(base_temp)

        # Compute positions
        total_width = base_width + gen_width + spacing
        start_x = (page_width - total_width) / 2
        y = margin + 20

        # Draw base image and box
        c.drawImage(base_temp, start_x, y, width=base_width, height=target_height)
        c.rect(start_x, y, base_width, target_height)  # rectangle
        c.drawCentredString(start_x + base_width / 2, y - 10, base_name)
        os.remove(base_temp)

        # Draw generated image and box
        gen_x = start_x + base_width + spacing
        c.drawImage(gen_temp, gen_x, y, width=gen_width, height=target_height)
        c.rect(gen_x, y, gen_width, target_height)  # rectangle
        c.drawCentredString(gen_x + gen_width / 2, y - 10, image_name)
        os.remove(gen_temp)

        c.showPage()

    except Exception as e:
        print(f"Error processing {image_name}: {e}")

# === SAVE PDF ===
c.save()
print("PDF created successfully with boundaries and increased page width.")
