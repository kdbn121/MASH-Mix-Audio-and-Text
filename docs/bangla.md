# Bangla

How this package treats Bangla: the language identifiers, the script rules, the number system, transliteration, regional detection and the preparation applied to lyric text.

## The identifiers

Bangla is registered in several standards at once and they do not agree, so `mash_mix.lang.registry` keeps every identifier next to its canonical BCP 47 tag.

| Tag | Name | ISO 639-1 | ISO 639-2/B | ISO 639-2/T | ISO 639-3 | ISO 639-5 | UN M.49 | CLDR language | CLDR script |
|---|---|---|---|---|---|---|---|---|---|
| `bn` | Bangla (Bengali) | bn | ben | beng | ben | ben | none | beng | Beng |
| `bn-BD` | Bangla - Bangladesh | bn | ben | beng | ben | ben | 050 | bng | Beng |
| `bn-IN` | Bengali - India | bn | ben | beng | ben | ben | 356 | beng | Beng |

| Tag | Native name | Region | Direction | Aliases |
|---|---|---|---|---|
| `bn` | বাংলা (বাঙালি) | none | ltr | beng, ben, bangla, bangali, bengali, bang, bn-beng, bn-latn-beng |
| `bn-BD` | বাংলা - বাংলাদেশ (বাঙালি বাংলা) | BD | ltr | bn-bd, beng-bd, ben-bd, bangla-bd, bangladesh, bn-bd-beng |
| `bn-IN` | বাংলা - ভারত (বাঙালি) | IN | ltr | bn-in, beng-in, ben-in, bangla-in, bengali-india, bn-in-beng |

### Resolving a tag

`resolve()` takes any of those identifiers, in any case, and returns the `LanguageSpec`. The lookup is case insensitive and treats an underscore as a hyphen.

```python
from mash_mix.lang import resolve

resolve("bn-BD")       # LanguageSpec(tag="bn-BD", ...)
resolve("bn_bd")       # LanguageSpec(tag="bn-BD", ...)
resolve("ben-bd")      # LanguageSpec(tag="bn-BD", ...)
resolve("bangladesh")  # LanguageSpec(tag="bn-BD", ...)
resolve("bng")         # LanguageSpec(tag="bn-BD", ...)
resolve("bangla")      # LanguageSpec(tag="bn", ...)
```

The resolution as it stands today:

| Input | Resolves to | Why |
|---|---|---|
| `bn-BD`, `bn_bd`, `ben-bd`, `beng-bd`, `bangladesh`, `bn-bd-beng` | `bn-BD` | explicit regional aliases |
| `bng` | `bn-BD` | the CLDR language code of the Bangladesh entry |
| `bn-IN`, `ben-in`, `beng-in`, `bangla-in`, `bengali-india` | `bn-IN` | explicit regional aliases |
| `bn`, `ben`, `beng` | `bn` | the shared ISO codes belong to the generic entry |
| `bangla`, `bangali`, `bengali`, `bang` | `bn` | aliases of the generic entry only |

The bare ISO codes are shared by all three entries, and the lookup gives them to the generic Bangla entry. Use `bn-BD` or `bn-IN` when the region matters.

`is_bangla()` is true for every registered tag and false for anything else. An unknown tag is not silently turned into Bangla: `resolve()` builds a generic specification for it, carrying the tag it was given and the note "Not registered in this package."

```python
from mash_mix.lang import is_bangla, list_languages, resolve, sibling_tags

is_bangla("bn-BD")                 # True
is_bangla("en")                    # False
resolve("bn-BD").region            # "BD"
resolve("zz").notes                # "Not registered in this package."
sibling_tags("bn-BD")             # ["bn-BD", "bn", "bn-IN"]
len(list_languages())              # 3
```

## Why CLDR has both beng and bng

CLDR treats Bengali and Bangla as two separate languages rather than one language with two regions.

- `beng` is Bengali. It is the older name and it covers the Indian form of the language, West Bengal, Tripura and Assam.
- `bng` is Bangla. It is the Bangladesh form, the one that carries the Eastern Nagari spelling conventions, the ৳ taka sign and the দাড়ি full stop.

They share a script, a grammar and most of their vocabulary, but they differ in spelling conventions, in numbers, and in everyday vocabulary. That is exactly the split this package models: `bn-IN` uses the CLDR code `beng` and `bn-BD` uses `bng`.

The practical consequence is that a locale-aware tool given `bng` produces Bangladeshi formatting, while the same tool given `beng` produces Indian formatting.

## The script

### Unicode block

| Range | Contents |
|---|---|
| `U+0980` to `U+09FF` | Bengali block, the script this package works with |
| `U+09E6` to `U+09EF` | Bangla digits ০১২৩৪৫৬৭৮৯ |
| `U+09BC` | NUKTA, the combining mark used by the nukta letters |
| `U+09F0` to `U+09F3` | Assamese digits, recognised but not used in output |

The language layer treats `U+0980` to `U+09FF` as Bangla when it measures the script mix of a string.

### Digits

Bangla digits map onto ASCII one to one in both directions.

| Bangla | ০ | ১ | ২ | ৩ | ৪ | ৫ | ৬ | ৭ | ৮ | ৯ |
|---|---|---|---|---|---|---|---|---|---|---|
| ASCII | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |

```python
from mash_mix.lang import normalize_bangla, to_ascii_digits, to_bangla_digits

normalize_bangla("123", digits="native")   # "১২৩"
normalize_bangla("১২৩", digits="ascii")    # "123"
to_bangla_digits("2026")                  # "২০২৬"
```

### Punctuation

The Bangla specific marks are the দাড়ি full stop `।` (U+0964), the visarga `ঃ` (U+0983), the taka sign `৳` (U+09F3) and the rupee sign `₹`. The full ASCII set is supported as well. `clean_asr_text` collapses repeated runs of `!?।,;:` down to a single mark.

### Nukta letters

Three Bangla letters carry a nukta: ড় (U+09DC), ঢ় (U+09DD) and য় (U+09DF). They are canonically decomposed into the base letter plus NUKTA (U+09BC), which is the form most Bangla tokenisers expect.

Unicode puts these three characters in the composition exclusion table, so NFC always rewrites the precomposed spelling into base plus NUKTA and never puts it back together. The `compose` setting of `normalize_bangla` therefore cannot produce the precomposed spelling.

| Precomposed | Decomposed |
|---|---|
| ড় | ড় |
| ঢ় | ঢ় |
| য় | য় |

```python
from mash_mix.lang import canonical_nukta, normalize_bangla

canonical_nukta("ড় ঢ় য়")   # "ড় ঢ় য়"
normalize_bangla("ড়")           # "ড়", decomposed
```

Pass `nukta="compose"` to `normalize_bangla` to go the other way, or `nukta="keep"` to leave the text alone.

### ZWNJ and ZWJ

This is the rule that generic Unicode cleaners get wrong, and it matters for Bangla more than for most scripts.

- **ZWNJ (U+200C) is kept.** It marks a real boundary between letters, and removing it merges two different words.
- **ZWJ (U+200D) is removed.** Bangla conjuncts are inherent in the script, so ZWJ carries no meaning there.

The two spellings below hold the same letters and differ only by the ZWNJ.

| Spelling | Codepoints | Meaning |
|---|---|---|
| ব্যবস্থা | ব ্ ব স ্ ত া | the cluster স্+ত, one connected conjunct |
| ব্যবস্‌তা | ব ্ ব স ্ ZWNJ ত া | the same letters, with ত kept visible as a separate letter |

In the second spelling the ZWNJ stops the hasant from swallowing the ত, so the ত is written out in full. Stripping the ZWNJ collapses the two words into one, and dropping the ZWJ from any text is safe because Bangla does not use it.

```python
from mash_mix.lang import normalize_bangla, strip_invisible

normalize_bangla("ব্যবস্‌তা")      # "ব্যবস্‌তা", the ZWNJ survives
normalize_bangla("ক‍ক")             # "কক", the ZWJ is gone
strip_invisible("ব্যবস্‌তা")        # "ব্যবস্‌তা"
strip_invisible("ব্যবস্‌তা", keep_zwnj=False)   # "ব্যবস্তা"
```

ZWSP (U+200B) becomes a space. The word joiner (U+2060), the BOM (U+FEFF), the soft hyphen (U+00AD) and the bidi control marks are dropped.

## The normalisation pipeline

`normalize_bangla` runs the steps in this order. The order matters: the nukta rewrite happens after the Unicode form is fixed, and the invisible characters are cleared before the digits are rewritten, so a digit that was separated by a stray mark is still found.

| Step | Function | What it does |
|---|---|---|
| 1 | `normalize_unicode(text, form)` | Applies NFC by default. NFKC folds widths if you ask for it |
| 2 | `canonical_nukta(text)` | Rewrites ড়, ঢ়, য় as base plus NUKTA. `compose` does the opposite, `keep` leaves it alone |
| 3 | `strip_invisible(text, keep_zwnj)` | Keeps ZWNJ, drops ZWJ, the word joiner, the BOM, the soft hyphen and the bidi marks, turns ZWSP into a space |
| 4 | `to_bangla_digits` or `to_ascii_digits` | Only when `digits` is `native` or `ascii`. `keep` leaves the digits alone |
| 5 | `normalize_whitespace(text)` | Collapses runs of spaces and tabs, normalises line endings, trims each line and the whole text |

```python
from mash_mix.lang import normalize_bangla

normalize_bangla("ড়")           # "ড়", decomposed
normalize_bangla("১২৩", digits="native")  # "১২৩"
normalize_bangla("১২৩", digits="ascii")   # "123"
```

### Cleaning recognised text

`clean_asr_text` is the pass applied to recognition output. It removes bracket noise, collapses repeated punctuation and tidies whitespace.

```python
from mash_mix.lang import clean_asr_text

clean_asr_text("[আড়ালা] গান!!!")   # "গান!"
clean_asr_text("[music] hello")   # "hello"
```

The bracket noise list covers আড়ালা, আড়ালো, গান, সঙ্গীত, music, applause, রশম, শব্দ, অশ্রু, কান্না, ইনস্ট্রুমেন্ট, background music, instrumental, laughing, applauding, noise, সাউন্ড ইফেক্ট and শব্দের ইফেক্ট.

### Measuring the script mix

```python
from mash_mix.lang import bangla_ratio, script_profile

bangla_ratio("আমি ভালো আছি")            # 1.0
script_profile("আমি fine 123")
# {"bangla": 0.4286, "devanagari": 0.0, "latin": 0.5714, "other": 0.0}
```

`bangla_ratio` counts letters only and ignores digits and punctuation. `script_profile` splits the letters into Bangla, Devanagari, Latin and other.

## Numbers

Bangla groups by lakh and crore, not by thousand and million.

| Group | Value | Word |
|---|---|---|
| hundred | 100 | শো, joined to the digit word: একশো |
| thousand | 1,000 | হাজার |
| lakh | 100,000 | লক্ষ |
| crore | 10,000,000 | কোটি |

### A worked example

`number_to_bangla_words(12345678)` reads the number as one crore, twenty three lakh, forty five thousand, six hundred and seventy eight.

| Part | Digits | Words |
|---|---|---|
| crore | 1 | এক কোটি |
| lakh | 23 | তেইশ লক্ষ |
| thousand | 45 | পঁয়তাল্লিশ হাজার |
| hundred | 678 | ছয়শো আটাত্তর |

The full reading is:

```
এক কোটি তেইশ লক্ষ পঁয়তাল্লিশ হাজার ছয়শো আটাত্তর
```

Bangla does not read a number as "forty and two". It uses the traditional compound word, so the reader carries its own tables for the teens and for every compound tens value from twenty upwards. That is why 42 is বল্লিশ and not চল্লিশ দুই.

| Value | Words |
|---|---|
| 0 | শূন্য |
| 1 | এক |
| 7 | সাত |
| 10 | দশ |
| 11 | এগারো |
| 21 | একুশ |
| 42 | বল্লিশ |
| 99 | নবানব্বই |
| 100 | একশো |
| 101 | একশো এক |
| 678 | ছয়শো আটাত্তর |
| 1000 | এক হাজার |
| 100000 | এক লক্ষ |
| 10000000 | এক কোটি |
| -5 | ঋণাত্মক পাঁচ |
| 3.5 | তিন দশমিক পাঁচ |

Decimals are read digit by digit after the word `দশমিক`. Negative numbers are prefixed with `ঋণাত্মক`.

### Parsing

`parse_number` accepts ASCII digits, Bangla digits, thousands separators, a leading sign and a decimal point. It returns `None` for anything it cannot read.

```python
from mash_mix.lang import parse_number

parse_number("12345678")   # 12345678.0
parse_number("১২৩")        # 123.0
parse_number("12,345")     # 12345.0
parse_number("-3.5")       # -3.5
parse_number("abc")        # None
```

### Dates, times and currency

```python
from mash_mix.lang import format_currency, format_date_bangla, format_time_bangla

format_date_bangla(2026, 11, 20)                       # "বিশ নভেম্বর, দুই হাজার ছাব্বিশ"
format_date_bangla(2026, 1, 1, bengali_calendar=True)   # "এক বৈশাখ, দুই হাজার ছাব্বিশ"
format_time_bangla(13, 45)                              # "তেরো টা পঁয়তাল্লিশ মিনিট"
format_time_bangla(13, 45, am_pm=True)                  # "এক টা পঁয়তাল্লিশ মিনিট অপরাহ্ন"
format_currency(1200)                                   # "১,২০০ টাকা"
```

The Gregorian month names are জানুয়ারি, ফেব্রুয়ারি, মার্চ, এপ্রিল, মে, জুন, জুলাই, আগস্ট, সেপ্টেম্বর, অক্টোবর, নভেম্বর, ডিসেম্বর. The Bengali calendar months are বৈশাখ, জ্যৈষ্ঠ, আষাঢ়, শ্রাবণ, ভাদ্র, আশ্বিন, কার্তিক, অগ্রহায়ণ, পৌষ, মাঘ, ফাল্গুন, চৈত্র.

`format_currency` writes the amount in Bangla digits with a thousands separator and names the currency in words. The supported currencies are ৳ টাকা, ₹ রুপি, $ ডলার, € ইউরো and £ পাউন্ড.

### Symbol expansion

A speech engine reads `50%` as "fifty percent" in English and as nothing at all in Bangla, so `expand_symbols` spells the symbols out before the text reaches the model.

| Symbol | Words | Symbol | Words |
|---|---|---|---|
| `%` | শতাংশ | `&` | এবং |
| `+` | যোগ | `=` | সমান |
| `@` | অ্যাট | `#` | নম্বর |
| `$` | ডলার | `€` | ইউরো |
| `£` | পাউন্ড | `¥` | ইয়েন |
| `₹` | রুপি | `৳` | টাকা |
| `°` | ডিগ্রি | `©` | কপিরাইট |
| `®` | রেজিস্টার্ড ট্রেডমার্ক | `™` | ট্রেডমার্ক |
| `~` | প্রায় | `*` | তারকা |
| `_` | আন্ডারস্কোর | `\|` | পাইপ |
| `~=` | প্রায় সমান | `*=` | গুণ |
| `/=` | ভাগ করে | `->` | থেকে |
| `=>` | থেকে | | |

```python
from mash_mix.lang import expand_symbols

expand_symbols("50% A & B")   # "50 শতাংশ A এবং B"
```

### Text prepared for a speech engine

`normalize_for_tts` is the full preparation used by the text to speech task.

```python
from mash_mix.lang import normalize_for_tts

normalize_for_tts("50%")                 # "৫০ শতাংশ"
normalize_for_tts("A & B")                # "A এবং B"
```

| Step | What it does |
|---|---|
| 1 | `normalize_bangla` with NFC, nukta decomposition and ZWNJ removed by default |
| 2 | `expand_symbols` |
| 3 | Optional number expansion when `expand_numbers=True` |
| 4 | Digits rendered as Bangla digits for `bn-BD` and as ASCII digits otherwise |
| 5 | Whitespace collapsed |

## Transliteration

Romanised Bangla, sometimes called Banglish, is converted to Bangla script in two layers.

### The rule engine

The engine splits a word into units and keeps the vowel and consonant positions explicit.

Consonant digraphs, longest first:

| Roman | Bangla | Roman | Bangla | Roman | Bangla |
|---|---|---|---|---|---|
| chh | ছ | kh | খ | th | থ |
| shh | ষ | gh | ঘ | dh | ধ |
| ph | ফ | bh | ভ | jh | ঝ |
| ng | ঙ | ny | ঞ | ch | চ |
| sh | শ | ts | টস | zh | জ |
| cch | ছ | khh | খ | | |

Vowels: `a` অ, `aa` আ, `i` ই, `ii` ঈ, `u` উ, `uu` ঊ, `e` এ, `ee` ই, `o` ও, `oo` উ, `ai` ঐ, `au` ঔ, `oi` ওই, `ou` ওউ.

A consonant that follows another consonant takes a hasant, so the pair becomes a cluster. A word final `a` is the inherent vowel and adds nothing. A word final `o` becomes the o kar on the previous consonant.

### The exception dictionary

The rules cannot be trusted for the frequent words, so `lang.lexicon.KNOWN_BANGGLISH` holds 256 curated entries that are applied first.

| Roman | Bangla | Roman | Bangla |
|---|---|---|---|
| ami | আমি | tumi | তুমি |
| apni | আপনি | apne | আপ্নে |
| kemon | কেমন | bhala | ভালো |
| bhalobashi | ভালোবাসি | dhonnobad | ধন্যবাদ |
| ghor | ঘর | kotha | কথা |
| bangladesh | বাংলাদেশ | dhaka | ঢাকা |
| shonghar | সংগীর | gaan | গান |

### The limits

Romanised Bangla is ambiguous and a mechanical conversion is never a substitute for reading the sentence.

- **The spelling carries no vowels.** `ghor` can be ঘর or ঘোর. The dictionary resolves the common ones, and the rules guess the rest.
- **Every letter is mapped, so a wrong word still gets a reading.** `hello` becomes হেল্লো, which is a transliteration rather than a translation. The exception dictionary fixes the frequent words, but an unusual romanisation can produce a reading the writer never intended.
- **Cluster boundaries are guesses.** A word with three or more consonants in a row can be segmented more than one way.
- **Loanwords and mixed text are risky.** Code switched text such as `bhaja bhaja` becomes ভাজা ভাজা, which is not what anyone meant.
- **It is not a phonetic transcription.** `bangla_to_latin` in the other direction is a transliteration, not a pronunciation guide. ভালোবাসি comes back as `bhalobasi`.

```python
from mash_mix.lang import bangla_to_latin, banglish_to_bangla, has_banglish

banglish_to_bangla("ami tomay bhalobashi")   # "আমি তোমায় ভালোবাসি"
banglish_to_bangla("bangladesh dhaka")       # "বাংলাদেশ ঢাকা"
bangla_to_latin("আমি")                        # "ami"
has_banglish("ami bangladesh dhaka")          # True
has_banglish("আমি তোমায়")                     # False
```

`has_banglish` is a cheap heuristic. It is false for any text that contains a Bangla character, and for Latin text it is true when at least half of the words contain two or more consonants in a row. English text such as "hello world" passes that test, so treat it as a hint rather than a decision.

Bangla script input passes through the transliterator unchanged apart from normalisation:

```python
banglish_to_bangla("আমি তোমায়")   # "আমি তোমায়"
```

## Regional detection

Bangla is one language with two standard registers. The detector says which one it found, and shows its working, so a wrong call can be seen and corrected rather than guessed at.

### The method

`detect_region` is a weighted marker vote.

1. The Bangla ratio of the text decides the script label: `bangla` at 0.3 and above, `latin` below 0.1, `mixed` in between.
2. Every marker in `BD_MARKERS` and `IN_MARKERS` is counted in the text. Each match adds its weight, and a single marker is counted at most five times so a repeated word cannot dominate.
3. The side with the higher score wins. The confidence is the winning share of the total.
4. The confidence is then scaled by the amount of text: `confidence * (0.6 + 0.4 * min(1, len(text) / 200))`. A short sentence can never be fully confident.
5. If neither side has a marker the caller's default is kept and the confidence is zero.
6. The dialect cluster is detected as well and reported in the same result.

```python
from mash_mix.lang import detect_region

report = detect_region("ভাই, আপ্নে কেমন আছেন? টাকা পাঠাতে হবে।")
report.tag           # "bn-BD"
report.confidence    # 0.6265
report.bd_score      # 3.8
report.in_score      # 0.3
report.dialect       # "dhaka"
report.evidence      # ["BD:আপ্নে x1", "BD:ভাই x1", ...]
```

A neutral sentence keeps the default with a confidence of zero:

```python
detect_region("আমি বাড়িতে যাই।").tag          # "bn-BD", the default
detect_region("আমি বাড়িতে যাই।").confidence   # 0.0
```

### The Bangladeshi markers

| Marker | Weight | Marker | Weight |
|---|---|---|---|
| আপ্নে | 3.0 | অইছে | 2.5 |
| অইল | 2.5 | অইলো | 2.5 |
| অইলাম | 2.5 | অইছিল | 2.0 |
| হইছে | 2.0 | হইল | 1.5 |
| দাও | 1.5 | বলো | 1.5 |
| আসো | 1.5 | খাও | 1.5 |
| নাও | 1.2 | চাও | 1.2 |
| ফেলো | 1.2 | ভাইয়া | 1.2 |
| গেলাম | 1.0 | মেয়েটা | 1.0 |
| ছেলেটা | 1.0 | ওয়াও | 1.0 |
| করতে পারলে না | 0.8 | খুব ভালো | 0.4 |
| ভাই | 0.4 | টাকা | 0.4 |
| আছে না | 0.3 | অনেক | 0.2 |

### The Indian markers

| Marker | Weight | Marker | Weight |
|---|---|---|---|
| ছিলেন | 2.5 | করেছিলেন | 2.5 |
| আপনি | 2.0 | এসেছেন | 2.0 |
| করেছেন | 2.0 | দিন | 1.5 |
| বলুন | 1.5 | আসুন | 1.5 |
| খান | 1.2 | নিন | 1.2 |
| ফেলুন | 1.2 | চান | 0.8 |
| মেয়ে | 0.8 | ছেলে | 0.8 |
| পেশে | 0.5 | নেই | 0.4 |
| খুব ভালো | 0.4 | ভাই | 0.3 |

### The dialect clusters

Nine clusters cover the regional voices inside Bangladesh. `detect_dialect` returns the cluster, a confidence and the evidence.

| Cluster | Markers |
|---|---|
| sylheti | খাওয়া হইছে, হাবি, কেনা, অইল, গেলা, নাই, খুব, বসে |
| chattogram | অইছে, হইছে, কোন, অল্প, ছিল, খুব, না |
| dhaka | আপ্নে, ভাই, ওয়াও, অইছে, খুব |
| barishal | অইছে, নাই, অইল, খুব |
| rajshahi | আপ্নে, অইছে, কর, ভাই |
| khulna | ভাই, আছে, আপ্নে, খুব |
| rangpur | অইছে, করিল, খুব |
| mymensingh | আপ্নে, অইছে, ভাই |
| comilla | অইছে, হইছে, ভাই |

```python
from mash_mix.lang import detect_dialect

cluster, confidence, evidence = detect_dialect("খাওয়া হইছে হাবি কেনা নাই")
cluster        # "sylheti"
confidence     # 0.5391
```

### The stop words and honorifics

| List | Bangladeshi | Indian |
|---|---|---|
| Stop words | 287 entries | 180 entries |
| Honorifics | ভাই, ভাইয়া, আপ্নে, আপনি, ভাইয, বউ, চাচা, মামা, ছাই, কাকা, আব্বা, আম্মু, ভাবি, বাবা, মা, দাদু, নানা, ঠাকুমা, জ্যেঠু, জেঠু | দা, দি, ভাই, আপনি, মামা, কাকা, বাবা, মা, দাদু, নানা, ঠাকুমা, জেঠু |

`COMMON_WORDS_BD` holds 44 frequent words that appear in songs and speeches, useful as a seed for a hotword list.

### The districts

The dictionary carries 63 district names with their Bangla spelling, arranged by division.

**Barishal**

| District | Bangla |
|---|---|
| Barguna | বরগুনা |
| Barishal | বরিশাল |
| Bhola | not in the dictionary |
| Jhalokati | ঝালকাঠি |
| Patuakhali | পটুয়াখালী |
| Pirojpur | পিরোজপুর |

**Chattogram**

| District | Bangla |
|---|---|
| Bandarban | বান্দরবান |
| Brahmanbaria | ব্রাহ্মণবাড়িয়া |
| Chandpur | চাঁদপুর |
| Chattogram | চট্টগ্রাম |
| Cox's Bazar | কক্সবাজার |
| Feni | ফেনী |
| Khagrachhari | খাগড়াছড়ি |
| Lakshmipur | লক্ষ্মীপুর |
| Noakhali | নোয়াখালী |
| Rangamati | রাঙ্গামাটি |

**Dhaka**

| District | Bangla |
|---|---|
| Dhaka | ঢাকা |
| Faridpur | ফরিদপুর |
| Gazipur | গাজীপুর |
| Gopalganj | গোপালগঞ্জ |
| Kishoreganj | কিশোরগঞ্জ |
| Madaripur | মাদারীপুর |
| Manikganj | মানিকগঞ্জ |
| Munshiganj | মুন্সিগঞ্জ |
| Narayanganj | নারায়ণগঞ্জ |
| Narsingdi | নরসিংদী |
| Rajbari | রাজবাড়ী |
| Shariatpur | শরীয়তপুর |
| Tangail | টাঙ্গাইল |

**Khulna**

| District | Bangla |
|---|---|
| Bagerhat | বাগেরহাট |
| Chuadanga | চুয়াডাঙ্গা |
| Jashore | যশোর |
| Jhenaidah | ঝিনাইদহ |
| Khulna | খুলনা |
| Kushtia | কুষ্টিয়া |
| Magura | মাগুরা |
| Meherpur | মেহেরপুর |
| Narail | নড়াইল |
| Satkhira | সাতক্ষীরা |

**Mymensingh**

| District | Bangla |
|---|---|
| Jamalpur | জামালপুর |
| Mymensingh | ময়মনসিংহ |
| Netrakona | নেত্রকোণা |
| Sherpur | শেরপুর |

**Rajshahi**

| District | Bangla |
|---|---|
| Bogura | বগুড়া |
| Chapainawabganj | চাঁপাইনবাবগঞ্জ |
| Joypurhat | জয়পুরহাট |
| Naogaon | নওগাঁ |
| Natore | নাটোর |
| Pabna | পাবনা |
| Rajshahi | রাজশাহী |
| Sirajganj | সিরাজগঞ্জ |

**Rangpur**

| District | Bangla |
|---|---|
| Dinajpur | দিনাজপুর |
| Gaibandha | গাইবান্ধা |
| Kurigram | কুড়িগ্রাম |
| Lalmonirhat | লালমনিরহাট |
| Nilphamari | নীলফামারী |
| Panchagarh | পঞ্চগড় |
| Rangpur | রংপুর |
| Thakurgaon | ঠাকুরগাঁও |

**Sylhet**

| District | Bangla |
|---|---|
| Habiganj | হবিগঞ্জ |
| Moulvibazar | মৌলভীবাজার |
| Sunamganj | সুনামগঞ্জ |
| Sylhet | সিলেট |

Two names in the dictionary sit outside the division lists: `Barman` (বরম), the historical name for Brahmanbaria, and `Rajshahi` (রাজশাহী), the district that shares its name with the division. `Bhola` is not in the district dictionary, although ভোলা appears in the transliteration exception list.

## Lyric preparation

Two different preparations apply to Bangla text, and they happen at different ends of the pipeline.

### Before a model sees the text

When Bangla text is fed to a speech engine, `normalize_for_tts` runs first.

| Order | Step | Why |
|---|---|---|
| 1 | Unicode NFC | Puts the text in one canonical form so the model sees one byte sequence per sound |
| 2 | Nukta decomposition | The decomposed form is what the Bangla tokenisers were trained on |
| 3 | ZWNJ removed | A speech engine has no use for the joiner control and some mispronounce around it |
| 4 | Symbols expanded | `%` becomes শতাংশ and `&` becomes এবং, because a model cannot guess what they mean |
| 5 | Numbers optionally expanded | `expand_numbers=True` turns 2500 into words so the reader does not misplace the units |
| 6 | Digits localised | Bangla digits for `bn-BD`, ASCII digits otherwise, because that is what the listener expects to hear |
| 7 | Whitespace collapsed | Removes the stray spaces a recogniser leaves behind |

### Before the text is written to a subtitle file

The recognised text goes through a different pass, because a subtitle should look tidy rather than be speakable.

| Order | Step | Why |
|---|---|---|
| 1 | `clean_asr_text` | Removes bracket noise such as `[আড়ালা]`, collapses repeated punctuation and tidies whitespace |
| 2 | `normalize_bangla` with `digits="keep"` | Applied only when the text is mostly Bangla. Keeps the digits the recogniser produced, because a song can contain a year or a price |
| 3 | Repetition control | Consecutive identical lines beyond `max_repeats` are dropped, which kills the decoding loop a model falls into on a repeated sung phrase |
| 4 | Short line merging | A fragment shorter than 1.2 seconds is joined with the next line when the gap is at most 0.8 seconds |
| 5 | Whitespace collapse | Every space run inside a line becomes one space |
| 6 | Line wrapping | `wrap_text` breaks the line on word boundaries at 42 characters and 2 lines, splitting a single over long word by character |
| 7 | Numbering and timestamps | SRT numbers the cues from 1 and writes `HH:MM:SS,mmm` timestamps |

The result is what you see in the SRT file:

```
1
00:00:00,000 --> 00:00:02,500
প্রথম লাইন

2
00:00:02,500 --> 00:00:05,125
দ্বিতীয় লাইন
```

Bangla script is written as it is, with no transliteration and no romanisation, so a subtitle player shows the script the singer actually sang.

---

Copyright (c) 2026 MASH (Khaled Bhuiyan). MIT Licence. See LICENSE.
