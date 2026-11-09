"""Controlled vocabularies: free-text -> normalised labels (synonym lists, not the official ontologies).

Each label has a regular expression of synonyms. To link to official ontology IDs (UBERON for tissues,
MONDO for diseases) later, add an ID per label here - the rest of the pipeline stays unchanged."""
import re

TISSUE = {
    "breast": r"breast|mammary",
    "lung": r"\blungs?\b|pulmonary|bronch|alveol|airway",
    "brain": r"\bbrain|cortex|cortical|hippocamp|cerebell|striatum|hypothalam|astrocyte|neuron|glioma|glioblastoma|spinal cord|neural",
    "blood": r"\bblood|pbmc|leuk[a]?emia|lymphocyte|monocyte|neutrophil|\bt cells?\b|\bb cells?\b|macrophage|\bplasma\b|\bserum|erythro|platelet",
    "bone_marrow": r"bone marrow|\bmarrow",
    "lymphoid": r"lymph node|spleen|thymus|tonsil|lymphoid",
    "heart": r"\bheart|cardiac|myocard|cardiomyocyte|ventric|atrial",
    "liver": r"\bliver|hepat",
    "colon": r"\bcolon|colorectal|rectum|rectal|\bbowel",
    "small_intestine": r"ileum|jejunum|duoden|small intestin",
    "skin": r"\bskin|melanoma|keratinocyte|dermal|dermis|epiderm|fibroblast",
    "kidney": r"kidney|renal|nephr",
    "prostate": r"prostat",
    "muscle": r"muscle|myotube|myoblast|myofiber",
    "pancreas": r"pancrea|islet",
    "ovary": r"\bovar",
    "uterus": r"\buter|endometri|myometri",
    "cervix": r"cervical|cervix",
    "testis": r"testis|testes|testic|sperm",
    "placenta": r"placent|trophoblast",
    "stomach": r"stomach|gastric",
    "esophagus": r"esophag|oesophag",
    "bladder": r"bladder|urothel",
    "thyroid": r"thyroid",
    "adipose": r"adipos|adipocyte|fat tissue|visceral fat",
    "bone": r"\bbone\b|osteo|chondro|cartilage",
    "eye": r"retina|retinal|\beye|cornea|ocular",
    "embryo_stem": r"embryo|blastocyst|stem cell|\bipsc|\bescs?\b|pluripotent",
    "plant": r"\bleaves|\bleaf|seedling|\broots?\b|\bshoots?\b|plant tissue|\bflower",
}

# cell lines imply a tissue of origin (used only when no tissue word was found)
CELL_LINE_TISSUE = {
    "cervix": r"\bhela\b", "breast": r"mcf-?7|mda-?mb|t47d|\bbt-?474", "liver": r"hepg2|huh-?7",
    "lung": r"\ba549\b|\bh1299|\bcalu-?3", "blood": r"jurkat|k-?562|\bthp-?1|\bhl-?60|\bu937",
    "kidney": r"hek ?293|\bhk-?2", "colon": r"caco-?2|hct-?116|ht-?29|sw480", "bone": r"\bu2os|saos-?2",
    "prostate": r"\blncap|\bpc-?3\b|du145", "skin": r"\ba375|hacat",
}

DISEASE = {
    "cancer": r"cancer|tumou?r|carcinoma|leuk[a]?emia|lymphoma|melanoma|glioma|glioblastoma|sarcoma|neoplas|metasta|myeloma|adenoma",
    "neurological": r"alzheimer|parkinson|schizophren|autism|neurodegener|epilep|huntington|dementia|multiple sclerosis|bipolar|depress",
    "cardiovascular": r"cardiovascular|cardiomyopath|atheroscler|heart failure|hypertension|myocardial infarction|ischemi|stroke",
    "immune_inflammatory": r"arthritis|lupus|autoimmun|inflammat|crohn|colitis|psoria|allerg|transplant rejection",
    "metabolic": r"diabet|obes|metabolic syndrome|insulin resist|fatty liver|nafld|\bnash\b",
    "infectious": r"infect|virus|viral|bacterial|\bhiv\b|tubercul|malaria|covid|sars-cov|influenza|hepatitis|sepsis",
    "respiratory": r"asthma|\bcopd\b|pulmonary fibrosis|cystic fibrosis|emphysema",
    "healthy": r"healthy|normal tissue|healthy donor|healthy volunteer|control subjects",
}

SPECIES_TEXT = {
    "human": r"\bhuman|homo sapiens", "mouse": r"\bmouse|\bmice\b|\bmurine|mus musculus",
    "rat": r"\brats?\b|rattus", "arabidopsis": r"arabidopsis", "yeast": r"yeast|saccharomyces|cerevisiae",
    "zebrafish": r"zebrafish|danio", "drosophila": r"drosophila|fruit fly", "e_coli": r"e\. ?coli|escherichia",
    "bacteria": r"bacteri|bacillus|staphylococ|streptococ|pseudomonas",
}
CELL_LINE = r"cell lines?|\bhela\b|hek ?293|mcf-?7|jurkat|k-?562|\ba549\b|hepg2|\bu2os|\bcos-?7|nih ?3t3|caco-?2|\bthp-?1"
# 'infect' deliberately removed (it is a disease word); keeps perturbation independent of the disease labels
TREATMENT = r"treated|treatment|stimulat|knockdown|knockout|sirna|shrna|crispr|inhibitor|\bdrug|compound|exposure|overexpress"
TIMECOURSE = r"time ?course|time ?series|\bhours?\b|\bdays?\b after"


def flag(text, pat):
    return "yes" if re.search(pat, text) else "no"


def first_match(text, patterns):
    """Label whose synonym appears earliest in the text (None if no label matches)."""
    best, pos = None, 10**9
    for label, pat in patterns.items():
        m = re.search(pat, text)
        if m and m.start() < pos:
            best, pos = label, m.start()
    return best
