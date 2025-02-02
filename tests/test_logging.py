import logging

logger = logging.getLogger(__name__)


def test_emoji():
    logger.info("🎉")


def test_emoji_via_shortcode():
    logger.info(":tada:")
