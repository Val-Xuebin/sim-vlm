import json

from PIL import Image

from vlm_sim.backends import MetadataBackend


def test_metadata_backend_reports_visible_types():
    image = Image.new("RGB", (64, 48))
    metadata = {"visibleObjects": [{"objectType": "Mug"}, {"objectType": "Mug"}]}
    result = json.loads(MetadataBackend(metadata).describe(image, "test"))
    assert result["objects"] == ["Mug"]
    assert "no pixel inference" in result["note"]
