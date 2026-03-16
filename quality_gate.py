import torch
import math
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class PageQualityGate:
    def __init__(self, model_id='distilgpt2', device=None):
        # 1. Load the Brain (distilgpt2)
        # It's smart enough to know names like "Anjali" or "Mumbai" are valid words.
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading AI Model on {self.device}...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        self.max_length = self.model.config.n_positions

    def score_page(self, text):
        """
        Takes an ENTIRE page of text, handles the length automatically,
        and returns a single quality score.
        """
        if not text or len(text.strip()) == 0:
            return 9999.0

            # Tokenize the whole page at once
        encodings = self.tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids.to(self.device)

        # Sliding Window Logic:
        # We process 1024 tokens at a time, moving forward by 512 tokens.
        # This gives us context from the previous chunk so sentences aren't cut in half.
        stride = 512
        seq_len = input_ids.size(1)

        nlls = []
        prev_end_loc = 0

        # Loop through the whole page
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids_chunk = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_chunk.clone()
            target_ids[:, :-trg_len] = -100  # Ignore context we've already scored

            with torch.no_grad():
                outputs = self.model(input_ids_chunk, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        # Calculate final average score for the page
        total_nll = torch.stack(nlls).sum()
        ppl = math.exp(total_nll / end_loc)

        return ppl


# ==========================================
# PASTE YOUR FULL PAGE CONTENT BELOW
# ==========================================
PAGE_TEXT = """
[DIAGRAM 0]

rr


[DIAGRAM 2]

THE CALCUTTA LAW JOURNAL. , [VoL. 91.

. _ ORIGINAL CIVIL. -

Before Mr. Justice R. 8. Bachawat. : °
IN THE GOODS OF
. ATUL KRISHNA MAJUMDAR
- DECEASED.

Probate—Non-conicntious proccedings—Recacation of grant, grounds of—
Material defects in procecdings—Circumstances that bar application for
revocation. : :

Proof in common form—Bearing of Sections 275-283 of the Indian Succession
Act 1925 and Section 68 of the Indian Evidence Act—Non-citation of a
person entitled to be cited, if a sufficient- ground for revocation of grant
‘Effect of non-citation, summary revocation or further proof in solemn
form, , :

Absence of an affidavit of an attesting witness to support a petition for
grant of Probate docs not make the procecdings defective in substance pro-
vided the petition is otherwise in accordance with the provisions of Sections
275-283 of the Indian Succession Act 1925.

Provisions of Scction 68 of the Indian Evidence Act are not exhaustive.
Under the Succession Act and the Rules of this High Court, declaration and
verification of the exccutor and an attesting witness is sufficient proof of the
Will, and the Court has the power to grant probate in non-contentious pro-
ceedings though there be no affidavit from the executor or the attesting
witnesses. : . ;

Proof of non-citation of «a person who ought to have been cited makes
the proceedings defective in substance but it is not in itself sufficient for
summary revocation of the probate granted ex-parte. The Court ought to
give the grantee an opportunity to prove the will in solemn form, and after
hearing evidence and objections, decide whether the order granting probate
should stand or whether it should be revoked. Proof of non-citation thus
converts a non-contentious proceedings into a defended action for proof in
solemn form.

Mere delay and acquiescence under circumstances not amounting to
waiver or estoppel does not bar the right of a person adverscly affected to
apply for revocation of grant of probate in non-contcntious proceedings, and
the onus of establishing such cstoppel or waiver is on the grantee.

Petition by Sm. Sudhalata Ghose for revoking the grant of
probate of will of her father, Atul Krishna Majumdar and if
necessary, for an order directing the executors to prove the wil}
jn solemn form, . ,

“ .


[VoL


Civin,
—-.
1953-

wre}
January, £2.


[DIAGRAM 6]

[DIAGRAM 7]

[DIAGRAM 8]

[DIAGRAM 9]

[DIAGRAM 10]

[DIAGRAM 11]

[DIAGRAM 12]

ne



==================================================
--- PAGE 2 (ROBUST PIPELINE) ---

[DIAGRAM 0]

a


[DIAGRAM 2]

VoL. 91.] . oe HIGH COURT. © :

: , - . .

_ The material facts of the case will appear -from the judg-_
ment, : . ‘

A, G. Mitra for the Petitioner. - So, J

Sankar Banerjee and Subimal Roy for the Respondents.

-The judgment of the Court was as follows: — *

Bachawat, J.t—This matter raises interesting. questions of
probate practice. This ‘is a petition by Sreemati Sudhalata
Ghose for’revoking the grant of probate of will ‘of her father
Atul Krishna Majumdar and, if necessary, for an order directing
the executors ‘to prove the will in solemn form.

Atul Krishna died on the 12th September, 1947. He left
behind him surviving his. widow Harimati, two daughters,
Sudhalata and Santilata, two grand daughters by a pre-deceased “
son and several grand children by Santilata and a.predeceased
daughter Snehalata.. One Satish Chandra Majumdar, Nirode
Chandra Majumdar, Kanai Lal. Choudhury and Charu Chandra
Choudhury as executors presented to this Court a petition for
grant of probate of the will on the 22nd December, 1947. The
order for grant of probate was made on the 22nd January,1948
wthout issuing any. special citation. Later Satish and Nerode
renounced the executorship. In 1952 Harifati instituted a suit
for administration of the estate of the deceased against Kanai
and Gharu as executors. .

* Tam called. upon to decide on affidavits the following pre-
liminary points:

. (1) Has the petitioner locus standi to maintain this appll-

cation? _ -— ;
- (2) Is she in the circumstances of this case debarred from .

. making the application? , a
. (3) Was'the proceeding for the grant of probate defective.

- in’ substance, because (a) the~petition for grant

. Was not supported by-an affidavit of an attesting


any
wont
v


Civ.
, 3953-
—
‘In the goods of
Atul Krishna
. Majumdar
deceased.
oo
January, 12.


[DIAGRAM 6]

[DIAGRAM 7]

x


[DIAGRAM 9]

ee




"""
# ==========================================

if __name__ == "__main__":
    gate = PageQualityGate()

    print("-" * 50)
    print("Analyzing Full Page...")

    score = gate.score_page(PAGE_TEXT)

    print(f"PAGE SCORE: {score:.2f}")
    print("-" * 50)

    # Simple Routing Logic
    if score < 60:  # Valid English
        print("✅ ROUTE: MANUAL DATABASE (Text is clean)")
    elif score < 100:  # Readable but maybe some noise
        print("⚠️ ROUTE: MANUAL (with caution) or QUICK REVIEW")
    else:  # Gibberish / Broken OCR
        print("❌ ROUTE: AI REPAIR (Text is broken)")