import asyncio
from playwright.async_api import async_playwright

async def save_fullpage_pdf(url, output_file="website.pdf"):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")

        # Get the full scrollable height of the page
        height = await page.evaluate("document.body.scrollHeight")

        # Save as one long single-page PDF
        await page.pdf(
            path=output_file,
            width="1920px",             # match screen width
            height=f"{height}px",       # full page height
            print_background=True
        )

        await browser.close()
        print(f"✅ Full-page PDF saved as {output_file}")

if __name__ == "__main__":
    url = input("Enter URL: ")  # change this
    asyncio.run(save_fullpage_pdf(url, input("Enter output file name: ")))
