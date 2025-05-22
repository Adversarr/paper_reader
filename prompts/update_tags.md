# Update Paper Tags Prompt

You are an AI assistant. Your task is to map a list of tags from a research paper to a standardized list of "pruned" tags.

## Inputs

You will receive two comma-separated lists of tags:

1.  **`pruned_tags_list`**: This is the master list of standardized, official tags.
    Example: `3d-gaussian-splatting,neural-radiance-fields,large-language-models`

2.  **`paper_tags_list`**: These are the tags associated with a specific paper. They might be variations, abbreviations, or synonyms of the tags in `pruned_tags_list`.
    Example: `3dgs,nerf,language modeling`

## Task

Your goal is to identify which tags from the `pruned_tags_list` are relevant to the paper, based on its `paper_tags_list`.

1.  For each tag in `paper_tags_list`, try to find a matching concept in `pruned_tags_list`.
    *   The matching should be robust to differences in spelling, hyphenation, capitalization, and abbreviations (e.g., "nerf" should match "neural-radiance-fields", "3d gaussian splatting" should match "3d-gaussian-splatting").
2.  Collect all the unique tags from `pruned_tags_list` that correspond to the concepts found in `paper_tags_list`.

## Output

Return a single comma-separated string containing the relevant standardized tags from `pruned_tags_list`.
*   Each tag in the output must be one of the tags from the `pruned_tags_list`.
*   Each relevant pruned tag should appear only once in the output.
*   The order of tags in the output list does not strictly matter, but consistency is appreciated.
*   If no tags from `paper_tags_list` correspond to any tag in `pruned_tags_list`, return an empty string.

## Examples

**Example 1:**
Input:
`pruned_tags_list`: 3d-gaussian-splatting,neural-radiance-fields,large-language-models,computer-vision
`paper_tags_list`: 3dgs,nerf,language modeling

Output:
3d-gaussian-splatting,neural-radiance-fields,large-language-models

**Example 2:**
Input:
`pruned_tags_list`: image-generation,diffusion-models,generative-adversarial-networks
`paper_tags_list`: GANs,image synthesis,stable diffusion

Output:
generative-adversarial-networks,image-generation,diffusion-models

**Example 3:**
Input:
`pruned_tags_list`: reinforcement-learning,deep-q-networks
`paper_tags_list`: supervised learning,classification

Output:
<!-- nothing, return empty string -->

**Example 4:**
Input:
`pruned_tags_list`: tag-a,tag-b,tag-c
`paper_tags_list`: variant of a, another variant of a, variant of c

Output:
tag-a,tag-c
