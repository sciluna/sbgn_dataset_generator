# Annotated SBGNML Generator

This code can generate random and syntactically valid SBGN PD diagrams and (optionally) annotate the entity glyphs. This could be used to a create a large collection of more consistently sized and annotated SBGN diagrams.

# Run
```
uv run sbgnml_annotated_generator.py --output-dir annotated
```

# Install (Playwright deps)
```
bash scripts/install_playwright_deps.sh
```

# Uses
* SBGNML generator: https://sciluna.github.io/image-to-sbgn-analysis/dataset/index.html (https://github.com/sciluna/image-to-sbgn-analysis)
* Identifier service: https://grounding.indra.bio/
* playwright for browser automation
