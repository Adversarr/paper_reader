# Prune Tags

As an assistant, your task is to process a list of tags or keywords from research papers and merge any duplicated concepts into a standardized format.

## Input
You will receive a comma-separated list of tags that may contain:
- Different spellings of the same concept
- Abbreviations and full terms
- Variations with hyphens, underscores, or spaces
- Mixed capitalization

## Task
1. Identify groups of tags that represent the same concept
2. For each group, select or create a standardized representation:
    - Prefer full terms over abbreviations (unless the abbreviation is more commonly used)
    - Use consistent formatting (hyphenated, lowercase is recommended)
    - Select the most accurate and complete version

<!-- INCLUDE: prompts/tag_survey/_tag_requirement.md  -->

## Output

Return a cleaned, comma-separated list of the standardized tags with duplicates removed.

You should not include any additional text or explanation in your response.

## Examples

**Input:**

3dgs,3d-gaussioan-splatting,gaussian splatting,3d-gaussian-splatting

**Output:**

3d-gaussian-splatting


**Input:**

nerf,neural-radiance-fields,NeRF,neural radiance field


**Output:**

neural-radiance-fields

