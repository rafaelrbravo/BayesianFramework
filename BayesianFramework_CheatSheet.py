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
    "`BayesianFramework` provides a compact interface for hierarchical Bayesian "
    "modeling with NumPyro, including global parameters, optional correlated "
    "entry-level local parameters, MCMC training, held-out prediction, test-time "
    "local inference, scoring, reproducible RNG control, and persistence."
)

DEPENDENCIES = "jax, numpyro, numpy, cloudpickle"

CONVENTIONS = [
    ("`arg{}`", "dictionary whose values have no entry dimension; used for one entry's data, local parameters, or model output, and for global parameters or other unbatched named values."),
    ("`arg{[]}`", "dictionary containing arrays spanning multiple entries; returned model outputs may also include a leading posterior/sample dimension."),
    ("`arg[]`", "1D sequence."),
    ("`Fn(...) -> value`", "function argument or method together with the value it returns."),
    ("`A | B`", "either input/return type A or input/return type B, depending on behavior."),
    ("*Italic arguments*", "accept `None` as a valid input."),
]

FUNCTIONS = [
    (
        "BayesianFramework(modelDataFull{[]}, ModelFn(globalParams{}, localParams{}, data{}) -> modelOut{}, *localParamNames[]*, *GlobalParamFn(dataShapes{}) -> globalParams{}*, choleskyConcentration, *rngSeed*) -> BayesianFramework",
        "Creates a framework object. **modelDataFull** contains the full dataset. **ModelFn** evaluates one entry. **localParamNames** optionally defines standardized entry-level parameters; when multiple locals are present, their correlation is learned with an LKJ prior controlled by **choleskyConcentration**, a positive number. **GlobalParamFn** optionally receives a dictionary of full-data array shapes and defines model-facing global parameters. **rngSeed** is an optional nonnegative integer that initializes the persistent RNG state used by calls that do not provide their own seed. **Returns:** a new `BayesianFramework` object.",
    ),
    (
        "Train(trainIndices[], TrainLikelihoodFn(globalParams{}, localParams{[]}, data{[]}, modelOut{[]}), numWarmup, numSamples, num_chains, acceptProb, dense_mass, medianSamples, printSummary, *rngSeed*) -> MCMC",
        "Fits the entries selected by **trainIndices** using NUTS MCMC. **TrainLikelihoodFn** defines the likelihood from the globals, locals, selected data, and model output. **numWarmup**, **numSamples**, **num_chains**, and **medianSamples** are positive integers; **acceptProb** is a probability; **dense_mass** and **printSummary** are booleans. **rngSeed** is an optional nonnegative integer that overrides the persistent RNG for this call without advancing the persistent RNG state. **Returns:** the NumPyro `MCMC` object.",
    ),
    (
        "Test(testIndices[], *TestLikelihoodFn(globalParams{}, localParams{[]}, data{[]}, modelOut{[]})*, *nPosteriorSamples*, numWarmup, numSamples, num_chains, acceptProb, dense_mass, medianSamples, *nTrajectorySamples*, printSummary, *rngSeed*) -> modelOut{[]} | MCMC[]",
        "Evaluates the trained model on entries selected by **testIndices**. If **TestLikelihoodFn** is omitted, no test MCMC is run: globals are drawn from the training posterior, new locals are sampled from their population distribution, and **nTrajectorySamples** trajectories are generated. If **TestLikelihoodFn** is provided, test MCMC infers the test-entry locals conditional on globals from the training posterior; **nPosteriorSamples** is an optional positive integer controlling how many training-posterior global states are tested. **numWarmup**, **numSamples**, **num_chains**, and **medianSamples** are positive integers; **acceptProb** is a probability; **dense_mass** and **printSummary** are booleans. **nTrajectorySamples** is an optional positive integer controlling the number of generated trajectories. **rngSeed** is an optional nonnegative integer that overrides the persistent RNG for this call without advancing the persistent RNG state. **Returns:** sampled `modelOut{[]}` without MCMC, or a list of NumPyro `MCMC` objects with MCMC.",
    ),
    (
        "ScoreTrain(ScoreFn(modelOut{}, data{}) -> score | score{}) -> scores | scores{[]}",
        "Scores stored training predictions. **ScoreFn** evaluates one entry and may return either a single score or a score dictionary. A single score produces an array over posterior samples and training entries; a dictionary produces the same dictionary structure with each value vectorized over those dimensions.",
    ),
    (
        "ScoreTest(ScoreFn(modelOut{}, data{}) -> score | score{}) -> scores | scores{[]}",
        "Scores the most recent test predictions. **ScoreFn** evaluates one entry and may return either a single score or a score dictionary. A single score produces an array over posterior samples and test entries; a dictionary produces the same dictionary structure with each value vectorized over those dimensions.",
    ),
    (
        "GetTrainModelOutput() -> modelOut{[]}",
        "Retrieves stored training model outputs. **Returns:** training model outputs with posterior-sample and entry dimensions.",
    ),
    (
        "GetTestModelOutput() -> modelOut{[]}",
        "Retrieves stored outputs from the most recent test. **Returns:** test model outputs with sample and entry dimensions.",
    ),
    (
        "SetRng(rngSeed) -> None",
        "Sets or resets the persistent RNG state using **rngSeed**, a nonnegative integer. Subsequent `Train()` and `Test()` calls use and advance this state unless they receive their own **rngSeed**; a call-specific seed overrides but does not advance the persistent state.",
    ),
    (
        "GetRecord(print) -> record{}",
        "Collects framework metadata; **print** is a boolean controlling whether it is pretty-printed. **Returns:** a dictionary containing parameter names, train/test indices, MCMC settings, RNG state, and function names.",
    ),
    (
        "Save(fileName) -> None",
        "Serializes the complete framework object, including trained state and functions, to **fileName**, a file-path string, using `cloudpickle`. Load only trusted saved files.",
    ),
    (
        "BayesianFramework.Load(fileName) -> BayesianFramework",
        "Loads an object previously written by `Save()`; **fileName** is a file-path string. **Returns:** the loaded `BayesianFramework` object.",
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
        Paragraph("<b>Purpose:</b> " + _markup(PURPOSE) + "<br/><b>Dependencies:</b> " + _markup(DEPENDENCIES), purpose_style),
        Paragraph("Conventions", heading_style),
        Paragraph("<br/>".join(convention_lines), conventions_style),
        Paragraph("User-Facing Functions", heading_style),
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
