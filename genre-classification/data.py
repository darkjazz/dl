import json
import textdistance

class TagMerger:
    def __init__(self) -> None:
        self.merged_tags = {
            # Tags with Slashes
            "pop / rock": "pop rock",
            "r&b/swing": "r&b swing",
            "rock / pop": "rock pop",
            "pop/funk": "pop funk",
            "pop/rock": "pop rock",
            "house/trance": "house trance",

            # Tags with Typos or Variants
            "mowtown: soul": "motown soul",
            "r & b": "r&b",
            "alt.rock": "alternative rock",
            "alt. rock": "alternative rock",
            "hardock": "hard rock",
            "soul/r&b": "soul r&b",
            "rocknroll": "rock and roll",
            "musical/film": "musical film",
            "pop   1970s": "1970s pop",
            "pop 30s 40s": "1930s-1940s pop",

            # Redundant Variants
            "brit pop": "britpop",
            "europop": "euro pop",
            "electronica & dance": "electronica",
            "singer/songwriter": "singer-songwriter",
            "folk, world, & country": "folk world country",
            "synth pop": "synthpop",
            "psychedelic rock": "psychedelia",
            "traditional pop": "pop",

            # Unifying Terms with Minor Differences
            "new york punk": "punk",
            "early pop/rock": "early pop rock",
            "modern electric chicago blues": "electric chicago blues",
            "classic pop vocals": "classic pop",
            "progressive house, house, progressive trance": "progressive house",
            "rnb": "r&b",

            # Standardizing Punctuation and Spaces
            "neo prog": "neo-prog",
            "c 86": "c-86",
            "post punk revival": "post-punk revival",
            "lo fi": "lo-fi",
            "trip hop": "trip-hop",
            "shoegaze": "shoegaze",

            # Merging Ambiguous Tags
            "adult contemporary": "adult contemporary",
            "adult alternative": "adult contemporary",
            "new wave": "new wave",
            "new romantic": "new wave",
            "electro": "electropop",
            "electropop": "electropop",
            "indietronica": "indie electronic",
            "indie electronic": "indie electronic",
            "classical pop": "classical pop",
            "orchestral pop": "classical pop",
        }

    def merge_tag(self, tag):
        return self.merged_tags.get(tag, tag)

class TagProcessor:
    def __init__(self, tag_data, min_occurrences=5, similarity_threshold=0.8):
        """
        Initialize the TagProcessor.

        Parameters:
        - tag_data (dict): A dictionary of tags and their occurrences.
        - min_occurrences (int): Tags with fewer than this number of occurrences are considered for matching.
        - similarity_threshold (float): Minimum similarity ratio (0 to 1) to consider tags as a match.
        """
        self.tag_data = tag_data
        self.min_occurrences = min_occurrences
        self.similarity_threshold = similarity_threshold

    def _find_similar_tags(self, tag, tags):
        """Find tags similar to the given tag."""
        similar_tags = []
        for candidate in tags:
            similarity = textdistance.jaro_winkler.normalized_similarity(tag, candidate)
            if similarity >= self.similarity_threshold and tag != candidate:
                similar_tags.append((candidate, similarity))
        return similar_tags

    def cleanse_tags(self):
        """Cleanse and merge similar tags."""
        tags = list(self.tag_data.keys())
        merged_tags = {}

        for tag in tags:
            if tag in merged_tags:
                continue

            # Find similar tags
            similar_tags = self._find_similar_tags(tag, tags)
            for match, similarity in similar_tags:
                if match in self.tag_data:
                    # Merge occurrences
                    self.tag_data[tag] += self.tag_data.pop(match, 0)
                    merged_tags[match] = tag

        return self.tag_data

    def find_best_matches(self):
        """Find best matches for tags with fewer than `min_occurrences`."""
        rare_tags = {tag: count for tag, count in self.tag_data.items() if count < self.min_occurrences}
        tags = list(self.tag_data.keys())
        best_matches = {}

        for tag in rare_tags:
            best_match = None
            highest_similarity = 0

            for candidate in tags:
                similarity = textdistance.jaro_winkler.normalized_similarity(tag, candidate)
                if similarity > highest_similarity and similarity >= self.similarity_threshold and tag != candidate:
                    best_match = candidate
                    highest_similarity = similarity

            if best_match:
                best_matches[tag] = best_match

        return best_matches

    def process(self):
        """Perform cleansing and find best matches."""
        self.cleanse_tags()
        return self.find_best_matches()


# Load tag data from the uploaded file
with open("/mnt/data/tag_occ.json", "r") as file:
    tag_data = json.load(file)

# Initialize and process tags
processor = TagProcessor(tag_data, min_occurrences=10, similarity_threshold=0.85)
best_matches = processor.process()

# Output the results
print("Best Matches:", best_matches)
