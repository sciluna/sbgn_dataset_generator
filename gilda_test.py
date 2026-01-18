#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gilda",
# ]
# ///
import gilda

scored_matches = gilda.ground('glucose')
print(scored_matches)

scored_matches = gilda.ground('BRAF')
print(scored_matches)
