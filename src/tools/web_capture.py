import os
import asyncio
import logging
from playwright.async_api import async_playwright
from datetime import datetime

logger = logging.getLogger(__name__)

async def capture_website_screenshot(url: str, output_dir: str = "artifacts") -> str:
    """
    Captures a full-page screenshot of a website given its URL.
    Handles preloaders and scroll-triggered animations (like GSAP/ScrollTrigger).
    Returns the file path to the saved screenshot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_url = url.replace("https://", "").replace("http://", "").replace("/", "_").split("?")[0][:30]
    filename = f"{safe_url}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    logger.info(f"Starting website screenshot capture for URL: {url}")
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            )
            
            page = await context.new_page()
            
            logger.info(f"Navigating to {url}...")
            await page.goto(url, wait_until="networkidle", timeout=60000)
            
            # General Preloader Detection: Wait for fullscreen overlays with loading/preloader keywords to disappear
            logger.info("Detecting and waiting for any active fullscreen preloaders/overlays to disappear...")
            try:
                # Execute a generalized JavaScript function inside the browser context
                await page.evaluate("""
                    async () => {
                        const isOverlayVisible = () => {
                            const elements = document.querySelectorAll('*');
                            const vw = window.innerWidth;
                            const vh = window.innerHeight;
                            
                            for (const el of elements) {
                                try {
                                    const rect = el.getBoundingClientRect();
                                    // Check if element covers at least 90% of the viewport
                                    if (rect.width >= vw * 0.9 && rect.height >= vh * 0.9) {
                                        const style = window.getComputedStyle(el);
                                        const isPositioned = ['fixed', 'absolute'].includes(style.position);
                                        const isVisible = style.display !== 'none' && 
                                                          style.visibility !== 'hidden' && 
                                                          parseFloat(style.opacity) > 0.1;
                                        
                                        if (isVisible && isPositioned) {
                                            const zIndex = parseInt(style.zIndex, 10) || 0;
                                            const className = (el.className || '').toString().toLowerCase();
                                            const id = (el.id || '').toString().toLowerCase();
                                            
                                            // Check for common loading/preloader signatures
                                            const hasLoadingKeyword = className.includes('loader') || 
                                                                     className.includes('preloader') || 
                                                                     className.includes('loading') || 
                                                                     className.includes('splash') ||
                                                                     className.includes('transition') ||
                                                                     id.includes('loader') || 
                                                                     id.includes('preloader') || 
                                                                     id.includes('loading') || 
                                                                     id.includes('splash');
                                            
                                            // If it has high z-index and contains loading keywords, it's a preloader
                                            if (zIndex > 10 || hasLoadingKeyword) {
                                                return true;
                                            }
                                        }
                                    }
                                } catch (e) {
                                    // Ignore errors for individual elements (e.g. SVG sub-elements)
                                }
                            }
                            return false;
                        };
                        
                        // Check periodically up to 8 seconds
                        const maxWait = 8000;
                        const interval = 200;
                        let elapsed = 0;
                        
                        return new Promise((resolve) => {
                            const check = () => {
                                if (!isOverlayVisible() || elapsed >= maxWait) {
                                    resolve(true);
                                } else {
                                    elapsed += interval;
                                    setTimeout(check, interval);
                                }
                            };
                            check();
                        });
                    }
                """)
                logger.info("Generalized preloader check completed.")
            except Exception as pe:
                logger.warning(f"Error during generalized preloader detection: {pe}. Proceeding...")
            
            # Allow page-load transitions to settle
            await asyncio.sleep(2)
            
            # Trigger scroll-driven / ScrollTrigger animations by scrolling step-by-step
            logger.info("Executing programmatic scrolling to trigger animations...")
            try:
                # Scroll down in dynamic increments to handle infinite loading and scroll-triggered animations
                y = 0
                step = 450
                max_scroll = 15000  # Cap maximum height to prevent getting stuck in infinite scroll pages
                
                while True:
                    await page.evaluate(f"window.scrollTo(0, {y})")
                    await asyncio.sleep(0.15)  # Let animations trigger and render
                    
                    y += step
                    scroll_height = await page.evaluate("document.documentElement.scrollHeight")
                    if y >= scroll_height or y >= max_scroll:
                        break
                
                # Reach absolute bottom
                await page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
                await asyncio.sleep(1)
                
                # Scroll back to the top
                await page.evaluate("window.scrollTo(0, 0)")
                await asyncio.sleep(1)
                logger.info("Scrolling and animations triggering complete.")
            except Exception as se:
                logger.error(f"Error during stepped scrolling: {se}")
            
            # Capture full page screenshot
            await page.screenshot(path=filepath, full_page=True, type="png")
            logger.info(f"Successfully captured screenshot of {url} as {filepath}")
            
            await browser.close()
            return filepath
            
    except Exception as e:
        logger.error(f"Error in capture_website_screenshot: {e}")
        raise e

