"""Generate BayesianFramework_CheatSheet.pdf.

The cheat-sheet content is intentionally kept at the top of this file so this
source is useful as a readable reference even without generating the PDF.

Dependency:
    pip install reportlab

Run:
    python BayesianFramework_CheatSheet.py

The PDF is written beside this file. The generator uses only ReportLab's
built-in Helvetica/Courier PDF fonts, so it does not depend on system fonts.
It automatically chooses the largest body font that keeps the cheat sheet on
one US Letter page.
"""


# -----------------------------------------------------------------------------
# Cheat-sheet content
# -----------------------------------------------------------------------------

TITLE = "BayesianFramework Cheat Sheet"

PURPOSE = (
    "`BayesianFramework` makes it easier to build, train, and evaluate hierarchical Bayesian models for data "
    "containing multiple related entries. A model can contain parameters shared across all entries, called global "
    "parameters, as well as local parameters that vary between entries. `Train` uses training data to infer the "
    "global parameters, the local parameters of the training entries, and the relationships among local parameters. "
    "`Test` carries the inferred global parameters and local-parameter relationships forward to new entries, using "
    "them to infer new local parameters either from the training results alone or by incorporating observations from "
    "the new entries. `ScoreTrain` and `ScoreTest` evaluate how well the resulting model predictions match the data. "
    "`Save` preserves the inferred model state so that potentially expensive training or testing does not need to be "
    "repeated, while `Load` restores that state for further testing, scoring, or analysis. The user defines the "
    "mathematical model and likelihoods, while the framework handles the Bayesian inference and organization needed "
    "to train, test, score, save, and reuse the model."
)

DEPENDENCIES = "jax, numpyro, numpy"

INSTALLATION = (
    "Install in editable mode with `pip install -e \"/path/to/BayesianFramework\"` then import with "
    "`from BayesianFramework import BayesianFramework`. Editable installation means updates pulled into "
    "the repository are available without reinstalling. If using VS Code and the import is not recognized "
    "by autocomplete, add `\"python.analysis.extraPaths\": [\"/path/to/BayesianFramework\"]` to settings.json."
)

CONVENTIONS = [
    ("`arg{}`", "dictionary whose values have no entry dimension; used for one entry's data, local parameters, or model output, and for global parameters or other unbatched named values."),
    ("`arg{[]}`", "dictionary containing arrays spanning multiple entries or samples."),
    ("`arg[]`", "1D sequence."),
    ("`Fn(...) -> value`", "function argument or method together with the value it returns."),
    ("`A | B`", "either input/return type A or input/return type B, depending on behavior."),
    ("*Italic arguments*", "accept `None` as a valid input."),
]

FUNCTIONS = [
    (
        "BayesianFramework(modelDataFull{[]}, ModelFn(globalParams{}, localParams{}, data{}) -> modelOut{}, *localParamNames[]*, *GlobalParamFn(dataShapes{}) -> globalParams{}*, choleskyConcentration, *rngSeed*) -> BayesianFramework",
        "Creates a framework object. **modelDataFull** contains the full dataset. **ModelFn** evaluates one entry. "
        "**localParamNames** optionally defines standardized entry-level parameters; when multiple locals are present, "
        "their correlation is learned with an LKJ prior controlled by **choleskyConcentration**, a positive number. "
        "**GlobalParamFn** optionally receives a dictionary of full-data array shapes and defines model-facing global "
        "parameters. **rngSeed** is an optional nonnegative integer that initializes the persistent RNG state. "
        "Stochastic operations advance this state; supplying **rngSeed** to a later call resets it before that operation. "
        "**Returns:** a new `BayesianFramework` object.",
    ),
    (
        "Train(trainIndices[], TrainLikelihoodFn(globalParams{}, localParams{[]}, data{[]}, modelOut{[]}), numWarmup, numSamples, num_chains, acceptProb, dense_mass, medianSamples, printSummary, *rngSeed*, saveLocals, saveModelOut) -> MCMC",
        "Fits the entries selected by **trainIndices** using NUTS MCMC. **TrainLikelihoodFn** defines the likelihood from "
        "the globals, locals, selected data, and model output. **numWarmup**, **numSamples**, **num_chains**, and "
        "**medianSamples** control MCMC; **acceptProb** is a probability; **dense_mass**, **printSummary**, **saveLocals**, "
        "and **saveModelOut** are booleans. **rngSeed** optionally resets the persistent RNG state before this call. "
        "**saveLocals** stores transformed training-local posterior samples; **saveModelOut** caches model outputs. "
        "**Returns:** the NumPyro `MCMC` object.",
    ),
    (
        "Test(testIndices[], *TestLikelihoodFn(globalParams{}, localParams{[]}, data{[]}, modelOut{[]})*, *nGlobalSamples*, numSamples, num_chains, numWarmup, acceptProb, dense_mass, medianSamples, printSummary, *rngSeed*, saveModelOut) -> None | MCMC[]",
        "Evaluates the trained model on **testIndices**. **nGlobalSamples=None** uses the median training-global posterior "
        "state; a positive integer samples that many training-global states. Without **TestLikelihoodFn**, no test MCMC is "
        "run: **numSamples** local samples are drawn per global state and stored for later scoring. With **TestLikelihoodFn**, "
        "one test MCMC is run per global state; **numSamples** is samples per chain and **num_chains** is the chain count. "
        "**numWarmup**, **acceptProb**, **dense_mass**, and **medianSamples** apply only to MCMC testing. **printSummary** "
        "prints MCMC summaries only. **rngSeed** optionally resets the persistent RNG state before this call. "
        "**saveModelOut** optionally caches test model outputs. **Returns:** `None` without MCMC, otherwise a list of "
        "NumPyro `MCMC` objects.",
    ),
    (
        "ScoreTrain(ScoreFn(modelOut{}, data{}) -> score{}, *nSamples*, *rngSeed*) -> scores{[]}",
        "Scores training trajectories. **ScoreFn** evaluates one entry and may return a scalar, array, or dictionary. "
        "**nSamples=None** scores all available joint training-posterior samples; otherwise that many posterior samples are "
        "selected using **rngSeed** or the persistent RNG. Saved model outputs are used when available; otherwise outputs are "
        "regenerated from saved posterior parameters. **Returns:** scores with leading posterior-sample and training-entry dimensions.",
    ),
    (
        "ScoreTest(ScoreFn(modelOut{}, data{}) -> score{}, *nSamples*, *rngSeed*) -> scores{[]}",
        "Scores the most recent test state. **nSamples=None** scores all available local samples; otherwise that many local "
        "samples are selected for every stored global sample using **rngSeed** or the persistent RNG. Saved model outputs are "
        "used when available; otherwise outputs are regenerated from saved test globals and locals. **Returns:** scores with "
        "leading global-sample, local-sample, and test-entry dimensions.",
    ),
    (
        "Save(fileName, saveTrainLocals, saveTestLocals, saveTrainModelOut, saveTestModelOut, saveAllData) -> None",
        "Writes framework state to a compressed NumPy `.npz` archive. Training globals and training Cholesky samples are "
        "always saved. **saveTrainLocals** and **saveTestLocals** default to `True`; saved test locals include their associated "
        "test-global and Cholesky state. **saveTrainModelOut**, **saveTestModelOut**, and **saveAllData** default to `False`. "
        "Functions are not serialized.",
    ),
    (
        "BayesianFramework.Load(fileName, ModelFn, *GlobalParamFn*, *modelDataFull*) -> BayesianFramework",
        "Loads framework state from a `.npz` file. **ModelFn** and optional **GlobalParamFn** are supplied explicitly rather "
        "than deserialized; their names are checked against the saved metadata and mismatches produce warnings. "
        "**modelDataFull** must be supplied unless the file was saved with `saveAllData=True`. "
        "**Returns:** the reconstructed `BayesianFramework` object.",
    ),
]


# -----------------------------------------------------------------------------
# PDF generation
# -----------------------------------------------------------------------------

from pathlib import Path
import html
import re

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph
    from reportlab.lib.styles import ParagraphStyle
except ImportError as exc:
    raise SystemExit(
        "Generating the PDF requires ReportLab. Install it with: pip install reportlab"
    ) from exc


OUTPUT_FILE = Path(__file__).with_name("BayesianFramework_CheatSheet.pdf")
MIN_BODY_FONT = 7.0
MAX_BODY_FONT = 11.5
FONT_STEP = 0.1


def _markup(text):
    """Convert the tiny readable markup used above to ReportLab paragraph markup."""
    text = html.escape(text, quote=False).replace("->", "-&gt;")
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def _signature_markup(text):
    """Render signatures bold overall, with None-accepting arguments italicized."""
    text = html.escape(text, quote=False).replace("->", "-&gt;")
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def _build_pdf(body_font):
    """Build the PDF at one candidate font size and return its page count."""
    page_width, page_height = letter
    margin = 0.38 * inch
    header_height = 0.38 * inch
    usable_height = page_height - 2 * margin - header_height

    frame = Frame(
        margin,
        margin,
        page_width - 2 * margin,
        usable_height,
        leftPadding=4,
        rightPadding=4,
        topPadding=2,
        bottomPadding=2,
    )

    def draw_header(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawString(margin, page_height - margin - 10, TITLE)
        canvas.setStrokeColor(colors.HexColor("#888888"))
        canvas.setLineWidth(0.5)
        canvas.line(margin, page_height - margin - 16, page_width - margin, page_height - margin - 16)
        canvas.restoreState()

    purpose_style = ParagraphStyle(
        "Purpose",
        fontName="Helvetica",
        fontSize=body_font + 0.8,
        leading=(body_font + 0.8) * 1.18,
        spaceAfter=4,
    )
    heading_style = ParagraphStyle(
        "Heading",
        fontName="Helvetica-Bold",
        fontSize=body_font + 2.5,
        leading=(body_font + 2.5) * 1.12,
        spaceBefore=2.5,
        spaceAfter=4.0,
    )
    conventions_style = ParagraphStyle(
        "Conventions",
        fontName="Helvetica",
        fontSize=body_font,
        leading=body_font * 1.18,
        spaceAfter=2.5,
        backColor=colors.HexColor("#F4F4F4"),
        borderPadding=4,
    )
    entry_style = ParagraphStyle(
        "Entry",
        fontName="Helvetica",
        fontSize=body_font,
        leading=body_font * 1.17,
        spaceBefore=2.2,
        spaceAfter=1.6,
        leftIndent=14,
        firstLineIndent=-14,
    )

    convention_lines = [f"<b>{_markup(symbol)}</b> - {_markup(description)}" for symbol, description in CONVENTIONS]

    story = [
        Paragraph("<b>Purpose:</b> " + _markup(PURPOSE) + "<br/><b>Dependencies:</b> " + _markup(DEPENDENCIES) + "<br/><b>Installation:</b> " + _markup(INSTALLATION), purpose_style),
        Paragraph("Conventions", heading_style),
        Paragraph("<br/>".join(convention_lines), conventions_style),
        Paragraph("Main Functions", heading_style),
    ]

    for signature, description in FUNCTIONS:
        story.append(
            KeepTogether([
                Paragraph(f"<b>{_signature_markup(signature)}:</b>  {_markup(description)}", entry_style)
            ])
        )

    doc = BaseDocTemplate(
        str(OUTPUT_FILE),
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    )
    doc.addPageTemplates(PageTemplate(id="Main", frames=[frame], onPage=draw_header))
    doc.build(story)
    return doc.page


def generate_pdf():
    """Use the largest body font that keeps the cheat sheet on one page."""
    steps = round((MAX_BODY_FONT - MIN_BODY_FONT) / FONT_STEP)
    candidate_sizes = [round(MAX_BODY_FONT - i * FONT_STEP, 1) for i in range(steps + 1)]

    for body_font in candidate_sizes:
        if _build_pdf(body_font) == 1:
            print(f"Wrote {OUTPUT_FILE.name} at {body_font:.1f} pt body font.")
            return

    raise RuntimeError(
        f"Cheat sheet does not fit on one page even at {MIN_BODY_FONT:.1f} pt. "
        "Shorten the content or lower MIN_BODY_FONT."
    )


if __name__ == "__main__":
    generate_pdf()
