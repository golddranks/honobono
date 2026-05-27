from tools import common
from tools.core import char_extract_per_sentence

stream = char_extract_per_sentence(
    episode_text=common.read_episode(1),
    model="Qwen3-14B-Q8_0.gguf",
)
for i, entry in enumerate(stream, 1):
    print(f"\n[{i:03d}] {entry['sentence']}")
    for p in entry["persons"]:
        print(f"      {p['name']:<12} {p['role']}")
