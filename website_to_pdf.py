import asyncio
from playwright.async_api import async_playwright
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import portrait

url_downloads = [
    "https://github.com/ActuallyNull/AFib-Detection-with-Machine-Learning/commit/dd826a02d9ff6eb2c264ca8c7e47ab213a601adc"
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
