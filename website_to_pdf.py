import asyncio
from playwright.async_api import async_playwright
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import portrait

url_downloads = [
 "https://github.com/RayanZamzaoui/afib-arrhythmia-ml-model/commit/0d30bde6b681393bd86aee7086ab6ac0e394e691",
 "https://github.com/RayanZamzaoui/afib-arrhythmia-ml-model/commit/ac35ccb46cfd0e5034ec8a53ed2e3934ab55b5f3",
 "https://github.com/RayanZamzaoui/afib-arrhythmia-ml-model/commit/849e037155d33fbfc27d3bf79a79773d9f9e0ba0",
 "https://github.com/RayanZamzaoui/afib-arrhythmia-ml-model/commit/7a0765bd63470fd070405f0f56690af1dcbbbce6",
 "https://github.com/RayanZamzaoui/afib-arrhythmia-ml-model/commit/1b17f14fa27ce48826bf801aba0aefe63549e2d8",
 "https://github.com/RayanZamzaoui/afib-arrhythmia-ml-model/commit/3cad777940729e1c32c6e5d491023cf5f3ed17e3",
 "https://github.com/RayanZamzaoui/afib-arrhythmia-ml-model/commit/3cecf5975ed7537083e76b4e9bfbb5f32c56fe1e",
 "https://github.com/RayanZamzaoui/afib-arrhythmia-ml-model/commit/0269252aa68014aafb3f26fb1360bb2a38f5a3a1"   
]

async def save_fullpage_pdf(url, output_file="website.pdf"):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")

        # Take a full-page screenshot (PNG)
        screenshot_path = "temp.png"
        await page.screenshot(path=screenshot_path, full_page=True)
        await browser.close()

        # Open screenshot with PIL to get dimensions
        img = Image.open(screenshot_path)
        width, height = img.size

        # Create a PDF with the same dimensions as the image
        c = canvas.Canvas(output_file, pagesize=portrait((width, height)))
        c.drawImage(screenshot_path, 0, 0, width=width, height=height)
        c.save()

        print(f"✅ Saved as a single-page PDF: {output_file}")

if __name__ == "__main__":
    for url in url_downloads:
        asyncio.run(save_fullpage_pdf(url, "commit_files/"+input("Enter output file name: ")+".pdf"))
