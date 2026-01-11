wrote a wrokarount do approxiamte the origianl get events function

```python
from pypore_compat import File, HMMBoard


# Load pre-segmented events
f = File.from_json('Data/14418004-s04.json')
events = [[seg.mean for seg in event.segments] for event in f.events]
```