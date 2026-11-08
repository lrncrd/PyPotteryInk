"""Style references of the cloud workflow: the prompt each mode sends and the images it needs."""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cloud


def test_style_reference_prompt_tells_the_model_to_copy_only_the_style_of_image_2():
    prompt = cloud.build_prompt("fine", reference="style")
    assert "Image 2 is ONLY a style reference" in prompt
    assert "Image 3" not in prompt and "images 2 and 3" not in prompt
    assert "never copy its shapes" in prompt
    # same rules and style fragment as the other modes
    assert cloud.STYLES["fine"]["fragment"] in prompt
    assert cloud._SECTIONS_RULE in prompt


def test_each_reference_mode_has_its_own_prompt_and_the_frozen_ones_are_untouched():
    prompts = {m: cloud.build_prompt("fine", reference=m) for m in cloud.REFERENCE_MODES}
    assert len(set(prompts.values())) == len(cloud.REFERENCE_MODES)
    assert prompts["none"].startswith(cloud.BASE_PROMPT)
    assert prompts["pair"].startswith(cloud.REFERENCE_PROMPTS["pair"])
    assert "Images 2 and 3 are ONLY a style reference" in prompts["pair"]
    assert "Image 2 is ONLY a style reference sheet" in prompts["sheet"]


def test_style_reference_needs_exactly_one_image():
    img = Image.new("RGB", (2000, 1000), "white")
    prepared = cloud.prepare_references("style", [img])
    assert len(prepared) == 1 and max(prepared[0].size) == cloud.REFERENCE_LONG_SIDE["style"]
    with pytest.raises(ValueError):
        cloud.prepare_references("style", [img, img])
    with pytest.raises(ValueError):
        cloud.prepare_references("style", [])
