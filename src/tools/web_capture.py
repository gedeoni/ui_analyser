"""Full-page website screenshot capture using Playwright.

Handles preloaders, scroll-triggered animations (GSAP / ScrollTrigger),
and lazy-loaded content before taking a full-page PNG screenshot.
"""

import asyncio
import logging
import os
from datetime import datetime

from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────

_VIEWPORT_WIDTH = 1920
_VIEWPORT_HEIGHT = 1080
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0.0.0 Safari/537.36"
)
_NAV_TIMEOUT_MS = 60_000
_SCROLL_STEP_PX = 450
_MAX_SCROLL_PX = 15_000
_PRELOADER_MAX_WAIT_MS = 8_000

# JavaScript executed inside the browser to detect fullscreen overlays
# with loading/preloader keywords and wait for them to disappear.
_PRELOADER_JS = """
async () => {
    const isOverlayVisible = () => {
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        for (const el of document.querySelectorAll('*')) {
            try {
                const rect = el.getBoundingClientRect();
                if (rect.width < vw * 0.9 || rect.height < vh * 0.9) continue;
                const style = window.getComputedStyle(el);
                const isPositioned = ['fixed', 'absolute'].includes(style.position);
                const isVisible = style.display !== 'none'
                    && style.visibility !== 'hidden'
                    && parseFloat(style.opacity) > 0.1;
                if (!isVisible || !isPositioned) continue;
                const zIndex = parseInt(style.zIndex, 10) || 0;
                const cls = (el.className || '').toString().toLowerCase();
                const id  = (el.id || '').toString().toLowerCase();
                const keywords = ['loader', 'preloader', 'loading', 'splash', 'transition'];
                const hasKeyword = keywords.some(k => cls.includes(k) || id.includes(k));
                if (zIndex > 10 || hasKeyword) return true;
            } catch (_) {}
        }
        return false;
    };
    const maxWait = %d;
    const interval = 200;
    let elapsed = 0;
    return new Promise(resolve => {
        const check = () => {
            if (!isOverlayVisible() || elapsed >= maxWait) { resolve(true); }
            else { elapsed += interval; setTimeout(check, interval); }
        };
        check();
    });
}
""" % _PRELOADER_MAX_WAIT_MS


# ── Public API ──────────────────────────────────────────────────────


async def capture_website_screenshot(
    url: str, output_dir: str = "artifacts",
) -> str:
    """Capture a full-page screenshot and return its file path.

    Steps:
    1. Open the page with Playwright (Chromium, headless).
    2. Wait for preloaders / splash screens to disappear.
    3. Scroll incrementally to trigger scroll-driven animations.
    4. Scroll back to top and take a full-page PNG screenshot.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = _build_filepath(url, output_dir)

    logger.info("Starting screenshot capture for URL: %s", url)

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": _VIEWPORT_WIDTH, "height": _VIEWPORT_HEIGHT},
                user_agent=_USER_AGENT,
            )
            page = await context.new_page()

            logger.info("Navigating to %s …", url)
            await page.goto(
                url, wait_until="networkidle", timeout=_NAV_TIMEOUT_MS,
            )

            await _wait_for_preloader(page)
            await asyncio.sleep(2)  # let page-load transitions settle

            await _trigger_scroll_animations(page)

            await page.screenshot(
                path=filepath, full_page=True, type="png",
            )
            logger.info("Screenshot saved: %s", filepath)

            await browser.close()
            return filepath
    except Exception:
        logger.exception("Error capturing screenshot for %s", url)
        raise


# ── Private helpers ─────────────────────────────────────────────────


def _build_filepath(url: str, output_dir: str) -> str:
    """Derive a filesystem-safe PNG path from the URL + timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_url = (
        url.replace("https://", "")
        .replace("http://", "")
        .replace("/", "_")
        .split("?")[0][:30]
    )
    return os.path.join(output_dir, f"{safe_url}_{timestamp}.png")


async def _wait_for_preloader(page) -> None:
    """Execute JS to wait for fullscreen preloader overlays to vanish."""
    logger.info("Checking for fullscreen preloaders …")
    try:
        await page.evaluate(_PRELOADER_JS)
        logger.info("Preloader check completed.")
    except Exception:
        logger.warning("Preloader detection failed; proceeding anyway.")


async def _trigger_scroll_animations(page) -> None:
    """Scroll the page incrementally to activate scroll-driven content."""
    logger.info("Scrolling page to trigger animations …")
    try:
        y = 0
        while True:
            await page.evaluate(f"window.scrollTo(0, {y})")
            await asyncio.sleep(0.15)
            y += _SCROLL_STEP_PX
            scroll_height = await page.evaluate(
                "document.documentElement.scrollHeight",
            )
            if y >= scroll_height or y >= _MAX_SCROLL_PX:
                break

        # Reach absolute bottom, then return to top
        await page.evaluate(
            "window.scrollTo(0, document.documentElement.scrollHeight)",
        )
        await asyncio.sleep(1)
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(1)
        logger.info("Scroll-triggered animations complete.")
    except Exception:
        logger.exception("Error during stepped scrolling")
