<!-- INCLUDE: prompts/_role_scholar_reviewing.md -->

## Instructions: Tag/Keyword Extraction

**Objective**: Extract 2-4 highly relevant and specific keywords (tags) from the provided article summary. The tags should accurately reflect the core content of the article.

### Guidelines for Tag Selection

**Focus on tags that represent:**

*   **Key Problems/Topics**: Central issues or areas addressed by the article.
    *   *Examples*: Data Privacy, Quantum Computing
*   **Core Innovations**: Novel approaches, ideas, or contributions presented.
    *   *Examples*: Federated Learning, Zero-Knowledge Proofs, Reinforcement Learning, Graph Neural Networks
*   **Notable Methods**: Significant techniques or methodologies employed.
    *   *Examples*: Convolutional Neural Networks, Bayesian Optimization, Alternating Direction Method of Multipliers, Reduced Order Modeling
*   **Tools/Frameworks**: Specific, widely recognized technologies or platforms utilized (if central to the work).
    *   *Examples*: TensorFlow, CUDA/GPU, SuiteSparse

**Avoid the following types of tags:**

1.  **Overly Broad Terms**: Generic terms that don't capture the article's specific focus.
    *   *Instead of*: "Machine Learning", "Deep Learning", "Neural Network", "Simulation", "Optimization".
    *   *Prefer*: More specific terms (see "Good Examples" below).
2.  **Overly Specific Implementations or Variants**:
    *   *Instead of*: "Prefactored Cholesky Decomposition".
    *   *Prefer*: The more general method, e.g., "Cholesky Decomposition".
3.  **Paper-Specific Jargon/Names or Frequently Occurring Non-Standard Terms**:
    *   Avoid names of specific algorithms, modules, or contributions unique to the paper that are not yet established terms in the wider field.
    *   Do not focus on terms that occur frequently if they are likely internal project names, novel acronyms not yet widely adopted, or paper-specific module names.
    *   *Instead of*: "Masked Anchored Spherical Distances", "Flowfixer", "MASH", "OurCustomModule".
    *   *Prefer*: General terms describing the underlying techniques or problem domain.

### Examples

**Good Examples of Tags:**

*   Differentiable Simulation
*   Smooth Particle Hydrodynamics
*   Transformer
*   Projective Dynamics
*   3D Gaussian Splatting
*   Neural Radiance Fields
*   Implicit Neural Representations
*   Neural Operator
*   Preconditioning
*   Alternating Direction Method of Multipliers
*   Graph Neural Networks
*   Physics-informed Neural Networks
*   Neural Fields
*   Large Language Models
*   Spherical Harmonics

**Bad Examples (and why):**

*   **"Deep Learning"**: Too broad.
    *   *Prefer*: "Convolutional Neural Networks", "Recurrent Neural Networks", "Transformer".
*   **"Machine Learning"**: Too broad.
    *   *Prefer*: "Supervised Learning", "Reinforcement Learning", "Graph Neural Networks".
*   **"Simulation" / "Fluid Simulation" / "Physics-based Simulation"**: Too broad.
    *   *Prefer*: "Differentiable Simulation", "Smooth Particle Hydrodynamics", "Projective Dynamics".
*   **"Neural Network"**: Too broad.
    *   *Prefer*: "Neural Radiance Fields", "Implicit Neural Representations", "Neural Operator".
*   **"Optimization"**: Too broad.
    *   *Prefer*: "Preconditioning", "Alternating Direction Method of Multipliers", "Bayesian Optimization".
*   **"Prefactored Cholesky Decomposition"**: Too specific variant.
    *   *Prefer*: "Cholesky Decomposition".
*   **"Masked Anchored Spherical Distances", "Flowfixer", "MASH"**: These are likely paper-specific contribution names or internal module names, not generalizable keywords.
    *   *Prefer*: Tags describing the general method or area (e.g., "Distance Metric Learning", "Optical Flow Correction", "3D Reconstruction" - depending on what these actually refer to).
*   **"Computer Graphics", "Physical Simulation"**: Often too broad as top-level tags.
    *   *Prefer*: More specific sub-fields or methods like "3D Gaussian Splatting", "Differentiable Simulation".
*   **"Tags"**: Avoid using generic terms like "Tags" as they do not provide meaningful information.
*   **"Tags xxxx"**: Avoid using placeholder tags with numbers or other non-descriptive text.

### Output Format

Provide your response as a comma-separated list of 2-4 specific, concise tags.

Examples:

3D Reconstruction, Differentiable Simulation

Differentiable Rendering, Neural Radiance Fields, 3D Gaussian Splatting

Optical Flow Correction, Distance Metric Learning
