# Talha's prep notes: slides 1 to 6

Yeh file sirf tumhare liye hai, recording ke liye nahi. `SCRIPT.md` mein woh hai
jo bolna hai; yahan woh hai jo **samajhna** hai, taake agar sir beech mein rok
kar sawal kar le to tum apne lafzon mein jawab de sako.

Har slide ke liye chaar cheezein: slide keh kya rahi hai, woh matter kyun karti
hai, kaunse number galat nahi hone chahiye, aur sir kya pooch sakte hain.

---

## The one-sentence version of the whole project

**English:** Deep learning models that classify network traffic read only the
first few hundred bytes of each connection. Nobody ever measured how many bytes
are actually needed. We measured it, found the field reads five to seven times
too many, and built a classifier that reads 144 bytes instead of 512 for the
same accuracy.

**Roman Urdu:** Network traffic ko pehchanne wale models har connection ke sirf
shuru ke kuch sau bytes parhte hain. Yeh number kisi ne kabhi measure nahi kiya,
bas pichle paper se copy hota raha. Humne measure kiya: pata chala field zaroorat
se **paanch se saat guna zyada** parh raha hai. Phir humne aisa classifier banaya
jo 512 ki jagah sirf **144 bytes** parh kar wohi accuracy deta hai.

Agar sirf yeh yaad rahe, to bhi tum 70% sawalon ka jawab de sakte ho.

---

---

## Slide 1, title

**Kya karna hai:** Apna taaruf, project ka sawal, aur yeh batana ke kaunsa hissa
kiska hai. Phir Zain ki ek line.

**Why it matters:** Sir ke liye yeh confirm hota hai ke dono members present
hain. Brief mein likha hai ke sirf ek member wali video complete group
presentation nahi mani jayegi, is liye yeh line skip nahi karni.

**Dhyan rakho:** Roll number saaf bolna, "i232548", digits alag alag.

---

---

## Slide 2, the problem, and the four gaps

### The problem: what an observation budget is

#### Slide keh kya rahi hai

Jab koi deep learning model network traffic classify karta hai, woh poori
conversation nahi dekhta. Woh sirf **shuru ke B bytes** dekhta hai. Us B ko hum
**observation budget** kehte hain.

#### Asaan misaal

**Roman Urdu:** Socho tumhe kisi ki awaaz se pehchanna hai ke phone pe kon bol
raha hai. Sawal yeh hai: **kitni second sunna zaroori hai?** Ek second? Paanch?
Classifier bhi bilkul yehi kaam karta hai, bas awaaz ki jagah bytes hain. Jitne
zyada bytes parho, utna der lagti hai faisla karne mein, aur utni zyada memory
chahiye.

**English, if asked:** It is like identifying a book from its opening pages. The
question is how many pages you need before you can tell what kind of book it is.

#### Why this is not a small detail

Teen cheezein seedha budget se tay hoti hain:

| | Roman Urdu mein |
|---|---|
| **Detection latency** | 1024 bytes ka budget hai to 1024 bytes aane se pehle faisla ho hi nahi sakta. Yaani der. |
| **Memory** | Router ya firewall ko har chalti hui connection ke B bytes yaad rakhne parte hain. Hazaron connections hain to memory barh jati hai. |
| **Kitni jaldi action** | Block karna, slow karna, alert bhejna, sab isi prefix ka intezar karte hain. |

Is liye budget koi side setting nahi, yeh **system ka operating point** hai.

#### The number that makes the point

**Zero out of twelve.** Baarah recent papers (2025 aur 2026) dekhe. Budget 5 se
20 packets, ya 784 se 1024 bytes. **Kisi ek ne bhi measure nahi kiya.** Jahan
wajah di bhi hai, woh technical convenience hai, jaise "784 bytes ka square
image ban jata hai", na ke koi experiment.

#### Agar sir pooche

> **"To kya sab galat kar rahe hain?"**
> Galat nahi, **unexamined**. Unhone number inherit kiya hai. Hum keh rahe hain
> ke yeh number measure hona chahiye tha, aur jab humne measure kiya to pata
> chala kaafi chhota ho sakta tha.

> **"Tumhara contribution kya hai phir?"**
> Curve. Accuracy versus budget ka curve, jahan baqi sab kuch fixed ho. Woh curve
> kisi paper mein nahi hai.

---

### The research gap

#### Slide keh kya rahi hai

Literature mein chaar cheezein missing hain. Hum chaaron ko order mein address
karte hain.

| Gap | Roman Urdu mein, ek line |
|---|---|
| **Gap 1** | Budget measure nahi kiya gaya, bas pichle paper se copy hua. |
| **Gap 2** | Sirf overall accuracy report hoti hai, is se pata nahi chalta ke kaunsi class bilkul fail ho rahi hai. |
| **Gap 3** | Dataset ko "encrypted" keh dete hain, magar kisi ne naapa nahi ke waqai kitna hissa encrypted hai. |
| **Gap 4** | IP address aur port jaise identifier bytes nikale gaye ya nahi, yeh batate hi nahi. |

#### The number worth saying

Gap 4 pe: humare apne base papers mein se ek (P5) ne dikhaya ke agar yeh
identifier bytes **andar chhor do**, to accuracy **0.45 tak** zyada aa jati hai.
Yaani model traffic nahi, **addresses yaad kar raha hota hai**.

#### Agar sir pooche

> **"Gap 2 ka matlab kya hai, example do."**
> ISCX pe humara overall accuracy 0.4013 hai, lekin ek class `vpn_chat` ka recall
> sirf **0.042** hai. Yaani us class ko model taqreeban kabhi predict hi nahi
> karta. Overall number dekh kar yeh bilkul nazar nahi aata. Zain yeh slide 8 pe
> dikhayega.

---

---

## Slide 3, the padding confound, and the objectives

### Budget and padding are confounded

Yeh slide sabse important hai is half mein. Isay achhe se samajh lo.

#### Slide keh kya rahi hai

Network flows **chhote** hote hain. ISCX dataset ka **median** session **274
bytes** ka hai, USTC ka **474 bytes**. Median ka matlab: aadhe sessions is se
chhote hain. Ab agar budget 1024 hai, to baqi jagah **padding** se bhar di jati
hai, yaani khali zeroes.

#### Asaan misaal, yeh zaroor yaad rakho

**Roman Urdu:** Socho ek form hai jisme jawab ke liye 1000 character ka box hai,
lekin tumhara jawab sirf 274 character ka hai. Baqi 726 khali reh gaye.

Ab agar box **aur bara** kar do, to kya jawab bara ho gaya? **Nahi.** Sirf khali
jagah barh gayi.

Yehi network pe ho raha hai. Budget barhane se zyada traffic nahi milta, **zyada
padding milti hai.**

#### The trap

Aur yahan asli masla aata hai. Model us khali jagah ko dekh kar hi andaza laga
sakta hai. Yaani:

**Roman Urdu:** Model soch sakta hai "is type ka message aam tor pe chhota hota
hai, aur yeh wala chhota hai, to shayad yeh wohi hai". Yeh **content parhe
baghair** jawab dena hua. Yeh **length** se guess kar raha hai, payload se nahi.

**English, the key sentence:** A paper that concludes "1024 bytes works best" may
have measured a length classifier and called it a payload classifier.

#### Numbers on this slide

Yeh **mean padding fraction** hai, yaani aik average session ka kitna hissa
padding hai:

| Budget | USTC | ISCX |
|---|---|---|
| 256 B | 0.105 | 0.244 |
| **512 B** | **0.242** | **0.478** |
| 1500 B | 0.398 | 0.701 |

ISCX pe 512 bytes pe aadha session padding hai. 1500 pe **70 percent**.

#### Agar sir pooche

> **"Padding to normal cheez hai, har model karta hai."**
> Bilkul. Masla padding nahi, masla yeh hai ke **budget aur padding ek saath
> barhte hain**, is liye jab koi kehta hai "bara budget behtar hai" to pata nahi
> chalta woh extra payload ki wajah se behtar hai ya extra padding ki. Dono
> effects mix hain. Hum inhe alag karte hain, slide 5 aur 9 pe.

> **"Yeh padding fraction kaise nikala?"**
> Formula paper mein Equation 2 hai: `max(0, k - L) / k`, jahan k budget hai aur
> L session ki asli length. Phir sab sessions ka average.

---

### Objectives

#### Slide keh kya rahi hai

Paanch objectives, har ek ka jawab measurable hai. Rattna nahi hai, **samajhna**
hai:

1. **Measure karo** budget curve, aur saturation point dhoondo. Saturation ka
   matlab: woh sabse chhota budget jiske aage barhane se koi faida reliably nahi
   hota.
2. **Check karo ke yeh model ki property to nahi.** Yaani kya koi aur
   architecture zyada bytes use kar leta? Is liye transformer bhi chalaya.
3. **Padding ko todo** do hisson mein: representation ka faida, aur PAD symbol ka
   faida.
4. **Aisa classifier banao** jo sirf utna parhe jitni zaroorat ho, with proof ke
   woh aage dekh hi nahi sakta.
5. **Imaandari se report karo:** shortcut learning band karo, per-class results
   do, aur batao ke train/test split akela kitna farq daal raha hai.

#### Saturation, asaan lafzon mein

**Roman Urdu:** Jaise phone charge karte waqt ek point ke baad battery percent
barhna band ho jata hai. Saturation point wohi hai: us ke baad bytes parhne se
accuracy nahi barhti.

#### Scale numbers

Do public corpora, **385 hazaar sessions**, **31 classes**, har number ke peeche
3 se 8 seeds.

Seeds ka matlab: **Roman Urdu:** har experiment 3 baar alag random start se
chalaya, phir average aur variation dono report kiye. Taake "yeh to luck tha"
na kaha ja sake.

---

---

## Slide 4, the pipeline and the leakage gate

### The measurement pipeline

#### Slide keh kya rahi hai

Agar budget ko **measure** karna hai, to budget hi **ek waahid cheez** honi
chahiye jo badal rahi ho. Baqi sab same rehna chahiye.

#### Asaan misaal, yeh bohat achhi hai

**Roman Urdu:** Socho tumhe test karna hai ke plate ka size khane ke taste pe
asar daalta hai ya nahi. To tum **ek dafa** degh pakao ge, phir usi degh se alag
alag size ki plates mein nikalo ge.

Agar har plate ke liye **naya** khana pakao, to masala thora bohat change ho hi
jayega, aur phir pata nahi chalega ke farq plate ki wajah se hai ya pakane ki.

Hum bilkul yehi karte hain: **ek dafa** 2048-byte ka HDF5 store banate hain, aur
har budget us ka slice hai.

#### Technical version

Parsing, sessionization, deduplication aur anonymisation sab **ek dafa** hote
hain, budget banne se **pehle**. Is liye class balance aur session boundaries 256
aur 512 bytes ke beech drift kar hi nahi sakte. Diagram mein sirf dashed wala
hissa har budget pe repeat hota hai.

**Group-wise split:** ek capture file poori ki poori ya training mein hai ya test
mein, dono mein nahi. Yeh Zain slide 11 pe explain karega ke yeh itna important
kyun hai.

---

### The leakage gate

#### Slide keh kya rahi hai

Gap 4 ko log aam tor pe ek line likh kar handle karte hain: "identifiers were
removed". Humne isay aisi **shart** bana diya jo pipeline khud check karti hai.

#### Asaan misaal, yeh sabse achhi hai

**Roman Urdu:** Jaise exam se pehle invigilator check karta hai ke kisi ke paas
parchi to nahi. Agar parchi mil jaye to **paper cancel**, sirf report mein note
nahi karte.

Humara gate bilkul yeh hai. Agar identifier bytes se kuch bhi seekha ja sakta
hai, to pipeline **ruk jati hai**. Warning nahi deti, **band ho jati hai**.

#### The condition itself

Ek probe sirf anonymised identifier bytes pe train hota hai. Us ka score
**majority-class rate plus 0.1** se zyada nahi hona chahiye.

**Majority class rate ka matlab:** agar tum bina kuch dekhe hamesha sabse bari
class ka naam le do, to jitne percent sahi honge. Yaani **guessing ka score**.
Agar probe guessing se behtar hai, matlab usay kuch seekhne ko mila, matlab
anonymisation fail hui.

#### Teen probes

| Probe | Kaam |
|---|---|
| `id_positions` | **Gate khud.** Anonymise karne ke **baad** check karta hai. |
| `ident` | Yeh naapta hai ke agar identifiers **chhor dete** to kitna faida hota. Yaani parchi ki qeemat. |
| `ports` | Sirf port bytes chupa kar dekhta hai, taake port memorisation aur asli payload structure alag ho sakein. |

#### Agar sir pooche

> **"Anonymise ka matlab kya kiya?"**
> MAC aur IP addresses zero kar diye. Ports retain kiye, lekin un ka asar `ports`
> probe se alag se naapa.

> **"Gate pass hua?"**
> Dono corpora pe pass hua. Zain exact numbers slide 7 pe dega: USTC pe probe
> 0.1351 aur chance 0.1359, yaani chance se bhi **neeche**.

---

---

## Slide 5, padding decomposed, and the classifier

### Padding decomposed into five input modes

#### Slide keh kya rahi hai

Paanch tareeqe se model ko input diya, taake pata chale faida kis cheez se aa
raha hai.

| Mode | Kya diya |
|---|---|
| `zero` | Har byte ko aik number, byte/255. Padding bhi zero. Yeh baseline hai. |
| `embed` | Har byte value ka **learned embedding**, lekin padding ab bhi zero. |
| `mask` | Embedding, **plus** padding ke liye alag symbol (PAD). |
| `convmask` | PAD symbol, aur convolution padded positions ko ignore karti hai. |
| `length_balanced` | Classes ko aisa resample kiya ke length ka cue khatam ho jaye. |

#### Asaan misaal

**Roman Urdu:** Agar tum cake banate waqt **recipe aur oven dono** ek saath badal
do aur cake behtar aaye, to tumhe pata nahi chalega ke faida recipe ne diya ya
oven ne.

Literature yehi karti hai: representation aur padding symbol **dono** ek saath
badalti hai, phir poora faida "padding-aware input" ko de deti hai.

Hum ek ek kar ke badalte hain:
- `zero` se `embed` = sirf **representation** badli
- `embed` se `mask` = sirf **padding symbol** badla

Is tarah dono ka hissa alag alag nazar aata hai.

#### Embedding, asaan lafzon mein

**Roman Urdu:** `zero` mode mein byte 200 aur byte 201 model ke liye taqreeban
same hain, kyunki dono 0.78 aur 0.79 ban jate hain. Embedding mein har byte
value ko apna alag "matlab" milta hai, jaise dictionary mein har lafz ka alag
meaning hota hai. Is liye model exact byte values pehchan sakta hai.

#### Seed spread ka usool

Har farq ko **seed spread** ke against check kiya jata hai. Agar farq seed noise
se chhota hai, to hum usay finding nahi, **inconclusive** likhte hain. Yeh
imaandari wali baat hai aur sir ko achhi lagti hai.

---

### The causal early-exit classifier

Yeh slide technically sabse bhaari hai. Lekin idea simple hai.

#### The idea

Model mein **paanch jagah** classification heads lagaye: 64, 128, 256, 512 aur
1024 bytes pe. Chalate waqt, **pehle head pe ruk jao** jiska confidence threshold
cross kar le, aur us flow ko sirf utne bytes charge karo.

**Roman Urdu:** Jaise exam mein agar pehle do paragraph parh kar hi jawab samajh
aa gaya, to poora passage parhne ki zaroorat nahi. Lekin agar confusion hai, to
aage parho.

#### The catch, yeh samajhna zaroori hai

Yeh claim tabhi sach hai jab head number k **dekh hi na sake** byte k+1 ko.

**Roman Urdu:** Agar student keh raha hai "maine sirf pehle do paragraph parhe",
to yaqeeni banana parega ke baqi page waqai **dhaka hua** tha. Agar woh jhaank
sakta tha, to claim jhoota hai aur saving sirf kaghazi hai.

#### Isay kaise guarantee kiya

- **Causal convolutions:** padding sirf **left** taraf se hoti hai, is liye koi
  position apne se aage nahi dekh sakti.
- **`ChannelNorm1d`, na ke BatchNorm ya GroupNorm.** Yeh sabse important technical
  detail hai.

**Roman Urdu, yeh zaroor yaad rakho:** BatchNorm aur GroupNorm poori length ka
average lete hain. Matlab position 900 ka byte, position 10 ki value ko hila
sakta hai. Bas ek line badalne se poori causality tooth jati, aur accuracy ke
numbers **bilkul theek lagte rehte**. Isi liye is pe regression test lagaya hua
hai.

#### Proof aur test, dono

- **Proof:** paper mein induction se prove kiya hai, layer by layer.
- **Test:** ek byte ko position 100 pe badla, aur naapa ke woh change kitni door
  tak jata hai. Jawab: **505 bytes aage, aur zero peeche.** Jo exactly analytic
  figure hai.

Yeh demo pe tum khud chala kar dikhao ge.

#### Agar sir pooche

> **"505 kahan se aaya?"**
> Formula: `1 + 2 x (kernel_size - 1) x sum(dilations)`. Yahan
> `1 + 2 x 4 x (1+2+4+8+16+32) = 1 + 504 = 505`. Chhe blocks, har block mein do
> convolutions, kernel 5, dilations 1 se 32.

> **"Yeh 505 aur 512 saturation itne qareeb hain, kya yeh confound nahi?"**
> **Yeh sabse achha sawal hai aur hum ne khud isay limitation mein likha hai.**
> Jawab: transformer control. Transformer pe aisi koi receptive field limit nahi
> hai, aur woh bhi **usi packet budget** pe saturate hota hai. Is liye 505 wajah
> nahi ho sakti. Zain yeh slide 13 pe cover karega.

---

---

## Slide 6, implementation, and the live demonstration

### Implementation

#### Slide keh kya rahi hai

Pandrah modules banaye: pcap reader se le kar sessionization, store, slicer,
chaar model families, aur experiment layer tak. **385 hazaar sessions**, taqreeban
**55 GB** captures se.

#### The line worth landing

**Reproducibility yahan mechanical hai, aspirational nahi.**

**Roman Urdu:** Har table aur har figure `make_report.py` se **khud ba khud**
banta hai result JSON se. Matlab paper, poster ya is deck mein **koi number haath
se type nahi kiya gaya**. Yeh keh dena strong impression deta hai, aur demo mein
tum yeh dikhao ge bhi.

#### Teen cheezein jo keh deni hain

1. `make_report.py` har table generate karta hai, hand typing zero.
2. `study.sh` stage-resumable hai, yaani epoch 30 pe crash ho to wahin se chalta
   hai, shuru se nahi.
3. Paanch regression tests, jin mein se ek **jaan boojh kar ek cue chhupata hai**
   known byte depth pe, aur sweep se demand karta hai ke woh usay dhoonde.

**Woh aakhri wala samjho:** **Roman Urdu:** Hum khud ek nakli signal daal dete
hain, maan lo byte 300 pe. Phir test yeh check karta hai ke humari sweep us ko
300 pe pakarti hai ya nahi. Agar pakar le, matlab sweep sach mein kaam kar rahi
hai. Yeh **positive control** hai.

---

### The live demonstration

Yahan tum terminal chalao ge. Commands `DEMO_RUNSHEET.md` mein hain. Yahan sirf
**har command kya sabit karti hai**, taake tum confidence se bol sako.

| Command | Kya sabit hota hai, asaan lafzon mein |
|---|---|
| `measure_receptive_field.py` | Ek byte position 100 pe badla. Positions 100 se 604 hile. Matlab **505 aage, zero peeche**. Encoder aage dekh hi nahi sakta. |
| `test_models_causality.py` | Head ko sirf apna hissa do ya poora sequence do, **jawab bilkul same** aata hai, difference zero. Yehi "bytes read" ko sach banata hai. |
| `sed` on `models.py` | Woh aik line, `ChannelNorm1d`, jis pe poori guarantee tiki hui hai. |
| `jq` on the sweep JSON | 0.9731 at 512 B, 0.9564 at 784, 0.9367 at 1500. **Curve neeche ja raha hai.** |
| `jq` on `significance.json` | 2488 sessions 512 pe sahi aur 784 pe ghalat, against sirf 94 ulta. Matlab yeh luck nahi. |
| `sed` on `REPORT.md` | Yeh table usi JSON se bana hai. Is liye deck ka koi number haath se type nahi hua. |

#### Ek cheez jo pooch sakte hain

> **"Pehle block mein head@512B ka difference zero kyun nahi?"**
> Kyunki woh **hona hi chahiye**. Test bytes 256 ke baad perturb karta hai, aur
> 512-byte head ko bytes 256 se 512 **dekhne ka haq hai**. Test sirf un heads pe
> zero demand karta hai jinka budget perturbation se pehle khatam ho jata hai.
> Yeh galti nahi, design hai.

#### Demo rules

- **Training kabhi mat chalana.** USTC study 40 epochs hai, video mein khatam
  nahi hogi.
- Har command recording se **pehle ek dafa** chala lena, taake Python ka import
  time camera pe pause na lage.
- Agar kuch fail ho jaye, **rukna mat**. Bolo "yeh machine pe slow hai, result
  `results/ustc/REPORT.md` mein hai, main wahan dikhata hoon". Yeh cut se behtar
  lagta hai.

---

## Numbers you must not get wrong

Yeh chhe numbers tumhare half mein aate hain. Baqi sab Zain ke hain.

| Number | Kya hai |
|---|---|
| **274 / 474 bytes** | Median session: ISCX / USTC |
| **0.478** | ISCX ka mean padding fraction 512 bytes pe |
| **505 bytes** | Receptive field, measured aur analytic dono |
| **144 bytes** | Adaptive policy kitna parhti hai USTC pe |
| **12 studies, 0 measured** | Literature ka khulasa |
| **385 hazaar sessions** | Project ka scale |

**Bolne ka tareeqa:** "zero point nine seven three one", na ke "point nine
seven". Precision hi is project ka asal point hai, is liye numbers poore bolo.

---

## Three questions that could come at you

> **"Tumhara contribution naya architecture hai?"**
> Nahi, aur hum yeh clearly likhte hain. Contribution **measurement** hai. Humne
> budget ko measured variable banaya, padding ko do hisson mein tora, aur ek
> causal early-exit classifier banaya jiska byte saving **sach** hai, nominal
> nahi.

> **"Tumhara ISCX accuracy itna kam kyun hai, 0.40?"**
> Teen measured wajahat: hum **11 classes** report karte hain jahan doosre 6
> karte hain; humara split **group-wise** hai jo zyada mushkil hai, aur woh akela
> **0.3782** ka farq daalta hai; aur humne identifier bytes **pehli run se
> pehle** nikal diye. Hum yeh number chhupate nahi, slide 12 pe table mein saaf
> likha hai.

> **"Yeh practically kis kaam ka hai?"**
> Agar koi firewall ya IDS bana raha hai, to yeh batata hai ke usay har flow ke
> 1024 bytes rokhne ki zaroorat nahi. 144 bytes kaafi hain. Matlab **kam memory,
> jaldi faisla, aur wohi accuracy.**

---

## Night before

1. `SCRIPT.md` do dafa parho, apne lafzon mein.
2. Slide 3 ka form wala misaal aur slide 4 ka parchi wala misaal rattna, yeh dono
   har sawal mein kaam aayenge.
3. Demo ki saari commands ek dafa chala lo.
4. Slide 1 Zain ke saath ek dafa rehearse karo, kyunki us mein dono bolte hain.
5. Mic ka level check karo, OBS meter pe peaks taqreeban minus 12 dB pe.
