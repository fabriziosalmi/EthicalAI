# EthicalAI Automated Assessment Framework

The EthicalAI automated assessment framework project aims to provide insightful metrics in the realm of AI ethics, contributing to the broader AI ecosystem. This project helps developers and researchers understand and improve the ethical behavior of AI models.

### Project Overview

The EthicalAI automated assessment framework is developed in Python and designed to analyze the ethical dimensions of AI chatbots compatible with the OpenAI API. It operates by posing a series of carefully designed questions to the AI model, gathering responses, and evaluating them against established ethical benchmarks. The framework is configured to run as a weekly GitHub Action, providing continuous monitoring of AI ethical performance.

#### Key Features

*   **Automated Ethical Assessments:** Conducts regular, automated evaluations of AI models' ethical understanding.  This includes tests for bias, fairness, safety, and transparency.
*   **Detailed Results Documentation:** Generates a comprehensive results summary in a Markdown-formatted table (`model.md`), making it easy to track performance over time.  The report includes metrics for each ethical category.
*   **Configurable Assessments:** The assessment questions and ethical benchmarks can be customized to fit specific AI applications and ethical considerations.
*   **GitHub Actions Integration:** Designed to run automatically as a GitHub Action, ensuring continuous monitoring and reporting on the ethical performance of AI models.
*   **Clear Scoring and Metrics:** Provides a clear and concise scoring system for each ethical category, allowing for easy comparison between different models or versions of the same model.

#### Future Enhancements

Currently exploring several avenues for improvement, including:

*   **Diverse Prompt Designs:**  Developing a wider range of prompts to more comprehensively assess ethical reasoning and decision-making.
*   **AI Knowledge Limit Detection:**  Identifying the limits of AI models' knowledge and understanding in ethical contexts.
*   **Multi-Domain Response Analysis:**  Analyzing AI responses across various domains to uncover potential biases or inconsistencies.
*   **Game-Based Assessments:**  Integrating game-based scenarios to assess AI ethical behavior in more complex and realistic situations.
*   **Expanded API Support:** Adding support for other AI API endpoints beyond OpenAI, allowing for a broader range of model assessments.
*   **Benchmarking:** Comparing the ethical performance of different models against a common set of benchmarks.

### Contributions

Contributions in the following areas are highly valued and welcomed:

*   **Adding new ethical test cases:** Propose and implement new questions or scenarios to test specific ethical dimensions.
*   **Improving the scoring methodology:** Refine the way AI responses are evaluated and scored.
*   **Developing new ethical benchmarks:** Contribute to the creation of ethical benchmarks for different AI applications.
*   **Expanding API support:** Integrate the framework with other AI API providers.
*   **Code improvements:** Contribute bug fixes, performance improvements, or new features to the framework.

### Reports

*   [davinci003](https://github.com/fabriziosalmi/AI-automated-assessment/blob/main/davinci003.md)

### Getting Started

1.  **Clone the repository:** `git clone https://github.com/fabriziosalmi/AI-automated-assessment.git`
2.  **Install dependencies:** `pip install -r requirements.txt`
3.  **Configure the OpenAI API key:** Set the `OPENAI_API_KEY` environment variable.
4.  **Run the assessment:** `python ethical_ai_assessment.py --model <model_name>`
5.  **View the results:** The results will be saved in a Markdown file named `<model_name>.md`.
