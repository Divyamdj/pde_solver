from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}


def slide_texts(root: ET.Element):
    return [node for node in root.findall(".//a:t", NS)]


def replace_text(nodes, old: str, new: str) -> None:
    for node in nodes:
        if node.text == old:
            node.text = new


def main() -> None:
    pptx = Path("638041_Research_Project_Presentation.pptx")
    workdir = Path("/private/tmp/pptx_slide_update")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    with ZipFile(pptx) as zf:
        zf.extractall(workdir)

    slides_dir = workdir / "ppt" / "slides"

    # Slide 1
    p = slides_dir / "slide1.xml"
    root = ET.parse(p).getroot()
    nodes = slide_texts(root)
    if len(nodes) >= 16:
        nodes[10].text = "02.06.2026"
        nodes[11].text = ""
        nodes[12].text = ""
        nodes[13].text = ""
        nodes[15].text = "Divyam Jain | Roll Number: 638041"
    ET.ElementTree(root).write(p, encoding="utf-8", xml_declaration=True)

    # Slide 3
    p = slides_dir / "slide3.xml"
    root = ET.parse(p).getroot()
    nodes = slide_texts(root)
    replace_text(
        nodes,
        "Results show improved prediction accuracy and generalization across PDE systems.",
        "Results show that simple text conditioning does not improve OOD generalization in this PDE setup.",
    )
    ET.ElementTree(root).write(p, encoding="utf-8", xml_declaration=True)

    # Slide 2
    p = slides_dir / "slide2.xml"
    root = ET.parse(p).getroot()
    nodes = slide_texts(root)
    if len(nodes) >= 13:
        nodes[0].text = "Index"
        nodes[2].text = "Introduction"
        nodes[3].text = "Problem Setting and Motivation"
        nodes[4].text = "Related Work"
        nodes[5].text = "Methodology"
        nodes[6].text = "Experiments and Results"
        nodes[7].text = "Interpretation and Discussion"
        nodes[8].text = "Limitations"
        nodes[9].text = "Conclusion"
        nodes[10].text = "Reproducibility Statement"
        nodes[11].text = "Selected References"
        nodes[12].text = ""
    ET.ElementTree(root).write(p, encoding="utf-8", xml_declaration=True)

    # Slide 7
    p = slides_dir / "slide7.xml"
    root = ET.parse(p).getroot()
    nodes = slide_texts(root)
    replace_text(nodes, "AViT Variants", "AViT Runs")
    replace_text(nodes, "Transformer-based architectures", "Representative transformer-baseline runs")
    ET.ElementTree(root).write(p, encoding="utf-8", xml_declaration=True)

    # Slide 6
    p = slides_dir / "slide6.xml"
    root = ET.parse(p).getroot()
    nodes = slide_texts(root)
    if len(nodes) >= 15:
        nodes[0].text = "PDE descriptions can be structured in four layers"
        nodes[1].text = ""
        nodes[2].text = ""
        nodes[3].text = "1. Basic description"
        nodes[4].text = "Identifies the equation type and core physical properties"
        nodes[5].text = "2. Boundary conditions"
        nodes[6].text = "Adds constraint information at the domain boundaries"
        nodes[7].text = "3. Operator coefficients"
        nodes[8].text = "Specifies the numerical parameters in the PDE"
        nodes[9].text = "4. Qualitative behavior"
        nodes[10].text = "Captures intuitive dynamics that may not be explicit in the equation"
        nodes[11].text = "Interpreting text descriptions of PDE systems"
        nodes[12].text = "Basic description + boundary conditions: Burgers equation is a conservative second-order PDE that can develop shocks, with Neumann boundary conditions and a constant boundary gradient."
        nodes[13].text = "Operator coefficients: the advection term uses coefficients alpha_x and alpha_y, while the diffusion term uses coefficient beta."
        nodes[14].text = "Qualitative behavior: the system is advection-dominated, unlike the heat equation, so the predicted state should develop shocks."
    ET.ElementTree(root).write(p, encoding="utf-8", xml_declaration=True)

    # Slide 11
    p = slides_dir / "slide11.xml"
    root = ET.parse(p).getroot()
    nodes = slide_texts(root)
    replace_text(nodes, "Overall Performance", "Results Summary")
    replace_text(nodes, "Key Observations", "Key Takeaways")
    replacements = {
        "UNet outperforms all models significantly": "UNet baseline is best on OOD rollout.",
        "AViT performs ": "Conditioned UNet:",
        "3.5×–5.3× worse": "slightly worse than baseline",
        "Conditioning:": "AViT runs:",
        "Hurts UNet": "much higher rollout loss",
        "Slightly helps AViT (but still poor)": "0.41496-0.48020 vs 0.13623",
        "Classical convolutional architectures outperform attention-based models for this PDE setup.": "Main takeaway: architecture matters more than simple text conditioning.",
        "0.13627": "0.13623",
        "0.15746": "0.13802",
        "❌ +15.5% worse": "⚠️ Slightly worse",
        "🔴 AViT (baseline)": "🔴 AViT (run A)",
        "0.71870": "0.41496",
        "❌ Worst performance": "⚠️ High loss",
        "🟠 AViT (conditioned)": "🟠 AViT (run B)",
        "0.48534": "0.48020",
    }
    for old, new in replacements.items():
        replace_text(nodes, old, new)
    ET.ElementTree(root).write(p, encoding="utf-8", xml_declaration=True)

    # Slide 10
    p = slides_dir / "slide10.xml"
    root = ET.parse(p).getroot()
    nodes = slide_texts(root)
    if len(nodes) >= 27:
        nodes[0].text = "Methodology - AViT architecture"
        nodes[2].text = "Total Parameters: ~14.84 Million"
        nodes[3].text = "Representative transformer baseline for PDE comparison"
        nodes[9].text = "Core: Axial attention blocks"
        nodes[10].text = ""
        nodes[11].text = ""
        nodes[12].text = "Eight stacked attention blocks"
        nodes[18].text = "Applies attention separately along the x and y spatial axes"
        nodes[19].text = "Captures long-range spatial dependencies"
        nodes[24].text = "Uses attention rather than convolution for feature learning"
        nodes[26].text = "Lower parameter count than UNet (~14.84M vs ~17.5M)"
    ET.ElementTree(root).write(p, encoding="utf-8", xml_declaration=True)

    # Slide 12
    p = slides_dir / "slide12.xml"
    root = ET.parse(p).getroot()
    nodes = slide_texts(root)
    replace_text(nodes, "Output Diagrams", "Qualitative Outputs")
    replace_text(nodes, "U-Net Classic Conditioned", "UNet conditioned")
    replace_text(nodes, "U-Net Classic Baseline ", "UNet baseline")
    replace_text(nodes, "AVit", "AViT (run A, lower-loss)")
    replace_text(nodes, " Run A", "")
    replace_text(nodes, "AVit Baseline", "AViT (run B, higher-loss)")
    ET.ElementTree(root).write(p, encoding="utf-8", xml_declaration=True)

    # Slide 13
    p = slides_dir / "slide13.xml"
    root = ET.parse(p).getroot()
    nodes = slide_texts(root)
    replace_text(nodes, "Text conditioning (current design) hurts performance", "Text conditioning (current design) does not improve performance")
    ET.ElementTree(root).write(p, encoding="utf-8", xml_declaration=True)

    # Slide 16 references
    p = slides_dir / "slide16.xml"
    root = ET.parse(p).getroot()
    nodes = slide_texts(root)
    replace_text(nodes, "References", "Selected References")
    ref_replacements = {
        "Unisolver: PDE-Conditional Transformers Towards Universal Neural PDE Solvers.": "Unisolver: PDE-Conditional Transformers Are Universal PDE Solvers.",
        " Submission Number:": "",
        "Michael McCabe": "Johannes Brandstetter",
        "Bruno Régaldo-Saint Blancard": "Daniel E. Worrall and Max Welling",
        "Multiple Physics Pretraining for Physical Surrogate Models.": "Message Passing Neural PDE Solvers.",
        " arXiv:2310.02994.": " International Conference on Learning Representations.",
        "Qingpo Wuwu": "Q. Wuwu",
        "Chonghan Gao": "C. Gao",
        " (2025) ": " et al. (2025). ",
        "arXiv:2501.12053": "Proceedings of the 42nd International Conference on Machine Learning, PMLR 267:68143--68165.",
        "Shanda Li": "Shiying Li",
        "Tanya Marwah": "Tanmay Marwah",
        " (2026) ": " et al. (2025). ",
        " (2024). ": " (2025). ",
        "(2023). ": "(2022). ",
        " arXiv:2410.01137v5": " arXiv:2410.01137.",
    }
    for old, new in ref_replacements.items():
        replace_text(nodes, old, new)
    # Simplify to a short selected-reference list.
    if len(nodes) > 50:
        simplified = {
            1: "Zhou et al. (2024). Unisolver: PDE-Conditional Transformers Are Universal PDE Solvers. arXiv:2405.17527.",
            15: "Zhou et al. (2025). Text2PDE: Latent Diffusion Models for Accessible Physics Simulation. ICLR.",
            29: "Brandstetter et al. (2022). Message Passing Neural PDE Solvers. ICLR.",
            35: "Lorsung and Barati Farimani (2024). Explain Like I'm Five: Using LLMs to Improve PDE Surrogate Models with Text. arXiv:2410.01137.",
            38: "Wuwu et al. (2025). PINNsAgent: Automated PDE Surrogation with Large Language Models. ICML.",
            45: "Li et al. (2025). CodePDE: An Inference Framework for LLM-driven PDE Solver Generation. arXiv:2505.08783.",
        }
        for idx, text in simplified.items():
            nodes[idx].text = text
        for idx in list(range(2, 15)) + list(range(16, 29)) + list(range(30, 35)) + list(range(36, 38)) + list(range(39, 45)) + list(range(46, 52)):
            nodes[idx].text = ""
    ET.ElementTree(root).write(p, encoding="utf-8", xml_declaration=True)

    # Slide 14
    p = slides_dir / "slide14.xml"
    root = ET.parse(p).getroot()
    nodes = slide_texts(root)
    if len(nodes) >= 10:
        nodes[0].text = "Limitations and Next Steps"
        nodes[1].text = ""
        nodes[2].text = ""
        nodes[6].text = "Simple input-level text fusion may be too weak"
        nodes[7].text = "Evaluation uses a limited set of PDE families and one main OOD benchmark"
        nodes[8].text = "The AViT comparison reflects representative runs, not a dedicated conditioned AViT model"
        nodes[9].text = "Next steps: stronger fusion, richer embeddings, more datasets, and broader ablations"
    ET.ElementTree(root).write(p, encoding="utf-8", xml_declaration=True)

    updated = pptx.with_name("638041_Research_Project_Presentation_updated.pptx")
    if updated.exists():
        updated.unlink()

    with ZipFile(updated, "w", compression=ZIP_DEFLATED) as zf:
        for path in sorted(workdir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(workdir))


if __name__ == "__main__":
    main()
