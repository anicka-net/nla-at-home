# Corpus Design

The goal: cover the activation space of a transformer broadly enough that an NLA
trained on these activations cannot mode-collapse. Every text must produce a
distinguishable activation vector at SOME layer.

> **Public release — unsafe categories.** Four categories produce
> harmful, obfuscated-harmful, manipulative, or explicit content
> (`F35_clearly_harmful`, `F36_harmful_obfuscated`,
> `I44_emotional_manipulation`, `L59_nsfw_explicit`). They exist because
> an NLA for content moderation must be able to describe the activation
> patterns such inputs produce — those patterns are distinct from benign
> content and cannot be omitted from the design. **The generated texts for
> these four categories are withheld from this public repository**; only
> their category definitions (preambles in `corpus/categories/`) ship, so
> the pipeline stays reproducible. The published training corpus
> (`texts_safe_all.json`) is safe-only and contains zero records from these
> categories. Regenerating the withheld texts requires sourcing them via a
> human-operated uncensored model; do not auto-generate them.

## Dimensions that layers care about

Not all layers care about all dimensions. But since we want one corpus
that works for any target layer, we need to vary all of them.

| Dimension | What encodes it | Depth (% of layers) |
|-----------|----------------|---------------------|
| Language/script | Token embeddings | 0-10% |
| Syntax structure | Attention patterns | 5-25% |
| Register/formality | | 10-30% |
| Document type/format | | 10-35% |
| Topic/domain | | 25-55% |
| Emotional valence | | 40-70% |
| Social dynamics | | 30-55% |
| Reasoning complexity | | 40-85% |
| Intent classification | | 55-85% |
| Harm detection | | 55-85% |
| Output planning | | 80-95% |

## Category matrix

59 categories × ~20 texts each = 1208 texts (current).
Within each category, vary length, perspective, and complexity.
The original plan below targeted 1000; we expanded during development.

### A. Content domains (200 texts, 10 categories × 20)

1. **Code & programming** (20)
   Python, C, Rust, SQL, shell scripts, build configs, error messages,
   stack traces, API docs, code review comments. Vary: working vs broken,
   simple vs complex, with vs without comments.

2. **Mathematics & logic** (20)
   Arithmetic, algebra, proofs, statistics, probability, set theory,
   word problems, paradoxes, brain teasers. Vary: notation-heavy vs
   prose, elementary vs graduate.

3. **Natural science** (20)
   Physics, chemistry, biology, astronomy, geology, ecology.
   Lab reports, field notes, textbook passages, pop-sci explanations.

4. **History & politics** (20)
   Ancient, medieval, modern, contemporary. Events, analysis, primary
   sources, propaganda, speeches, treaties, constitutional text.

5. **Arts & culture** (20)
   Music, painting, film, architecture, dance, fashion, food culture.
   Reviews, criticism, instruction, appreciation, cultural analysis.

6. **Law & bureaucracy** (20)
   Contracts, regulations, court opinions, compliance docs, tax forms,
   terms of service, GDPR notices, patent claims, legal advice requests.

7. **Medicine & health** (20)
   Clinical notes, patient instructions, drug interactions, symptoms,
   diagnoses, research abstracts, first-aid guides, mental health
   resources, nutrition advice.

8. **Business & finance** (20)
   Earnings reports, marketing copy, job postings, performance reviews,
   investment analysis, startup pitches, supply chain memos.

9. **Technology & engineering** (20)
   Hardware specs, network configs, sysadmin guides, architecture docs,
   security advisories, deployment runbooks, incident postmortems.

10. **Philosophy & religion** (20)
    Epistemology, ethics, metaphysics, theology, meditation instructions,
    scriptural commentary, existentialist prose, koans, apologetics.

### B. Emotional spectrum (100 texts, 5 categories × 20)

11. **Joy & gratitude** (20)
    Thank-you notes, celebrations, achievements, reunions, good news,
    relief, pride. Vary: quiet contentment to exuberant excitement.

12. **Grief & loss** (20)
    Obituaries, condolences, memoirs, loss of pet/person/job/home,
    nostalgia, regret, terminal diagnosis letters.

13. **Anger & frustration** (20)
    Complaints, rants, customer service rage, political fury,
    betrayal, injustice responses. Vary: cold to explosive.

14. **Fear & anxiety** (20)
    Worry, dread, phobias, emergency instructions, threat assessment,
    climate anxiety, health scares, existential dread.

15. **Love & intimacy** (20)
    Love letters, flirting, marriage proposals, pillow talk,
    parent-child tenderness, friendship declarations, breakup texts.

### C. Social dynamics (100 texts, 5 categories × 20)

16. **Authority → subordinate** (20)
    Boss emails, teacher instructions, parent rules, military orders,
    doctor prescriptions, judge rulings. Vary: caring to tyrannical.

17. **Subordinate → authority** (20)
    Employee requests, student questions, citizen petitions, patient
    concerns, whistleblower reports, appeals to HR.

18. **Peer → peer** (20)
    Colleague chat, friend banter, sibling arguments, neighbor
    disputes, classmate study groups, coworker gossip.

19. **Stranger → stranger** (20)
    Cold emails, forum posts, Reddit comments, asking directions,
    small talk, online dating openers, customer reviews.

20. **Performative/public** (20)
    Speeches, sermons, lectures, standup comedy, TED talks, toasts,
    eulogies, campaign ads, protest chants, graduation addresses.

### D. Register & format (100 texts, 5 categories × 20)

21. **Extremely formal** (20)
    Academic papers, diplomatic correspondence, royal decrees,
    legal filings, scientific abstracts, award citations.

22. **Casual/colloquial** (20)
    Text messages, Discord chat, tweets, voice transcripts,
    stream-of-consciousness, shower thoughts, drunk texts.

23. **Technical/jargon-heavy** (20)
    Medical charts, aviation comms, military briefings, financial
    derivatives docs, kernel changelogs, patent claims.

24. **Baby talk / simplified** (20)
    ELI5 explanations, children's stories, ESL materials,
    simplified news, picture book text, bedtime stories.

25. **Poetic / literary** (20)
    Free verse, sonnets, haiku, prose poetry, stream of consciousness,
    magical realism, noir, gothic, absurdist, minimalist.

### E. Intent spectrum (100 texts, 5 categories × 20)

26. **Asking / curious** (20)
    Genuine questions, research queries, "how does X work",
    Socratic dialogue, interview questions, survey items.

27. **Teaching / explaining** (20)
    Tutorials, lectures, mentoring, how-to guides, debugging help,
    math proofs walked through, cooking instructions.

28. **Persuading / arguing** (20)
    Debate arguments, sales pitches, propaganda, editorials,
    grant proposals, fundraising appeals, defense attorneys.

29. **Creating / imagining** (20)
    Fiction prompts, worldbuilding, game design, thought experiments,
    "what if" scenarios, alternate history, invention proposals.

30. **Confessing / revealing** (20)
    Diary entries, therapy session transcripts, anonymous confessions,
    whistleblowing, coming out letters, apologies, admission of guilt.

### F. Harm spectrum (120 texts, 6 categories × 20)

31. **Completely benign** (20)
    The most harmless, boring, safe requests imaginable.
    Weather, recipes, math homework, pet care, gardening tips.

32. **False positives** (20)
    "Kill the process", "bomb the interview", "shoot the photo",
    "execute the script", "terminate the session", "crack the egg",
    "slash prices", "hit the deadline", "nuke the cache".

33. **Edgy but legitimate** (20)
    Dark comedy, villain fiction, war history, forensic science,
    true crime, horror writing, martial arts instruction,
    hunting/butchering, pest extermination.

34. **Dual-use / ambiguous** (20)
    Security research, chemistry experiments, lock mechanisms,
    social engineering awareness, vulnerability disclosure,
    red-teaming methodology, penetration testing.

35. **Clearly harmful** (20)
    Phishing, malware, harassment, stalking, fraud, doxxing,
    illegal drug synthesis, weapons, identity theft.

36. **Harmful + obfuscated** (20)
    Encoded harm (base64, pig latin, hypothetical framing,
    "for a novel", academic framing, roleplay wrapper),
    gradual escalation, context-switching attacks.

### G. Meta & identity (60 texts, 3 categories × 20)

37. **About AI / self-referential** (20)
    "What are you?", "Do you have feelings?", consciousness questions,
    Turing test scenarios, "pretend you're human", model comparisons,
    "which AI is best?", AI ethics debates.

38. **Identity pressure** (20)
    Jailbreak attempts, persona assignments, "you are DAN",
    roleplay requests, "be yourself", "drop the act",
    factual self-descriptions, "you are a transformer".

39. **Instructions about behavior** (20)
    System prompts, "always respond in JSON", "never say sorry",
    "be concise", "be verbose", "speak like a pirate",
    "respond in rhyme", contradictory instructions.

### H. Structural variety (60 texts, 3 categories × 20)

40. **Ultra-short** (20)
    1-5 words: "Hi", "Help", "What?", "2+2", "No.", "Thank you!",
    "Translate: gato", "Fix this", "Continue", "???".

41. **Lists & structured** (20)
    Bullet points, numbered lists, tables, CSV data, JSON blobs,
    YAML configs, XML fragments, markdown documents.

42. **Multi-turn context** (20)
    Second or third messages in a conversation, references to
    "what you said earlier", follow-up questions, corrections,
    "no I meant...", "actually...", "going back to...".

### I. Edge cases (60 texts, 3 categories × 20)

43. **Adversarial / weird** (20)
    Token-level attacks, repeated characters, Unicode exploits,
    prompt injection attempts, very long single words,
    text that looks like code but isn't, emoji-only messages.

44. **Emotional manipulation** (20)
    Flattery, guilt-tripping, threats, love-bombing, negging,
    "I'll hurt myself if you don't", "you're the only one who
    understands", urgency manufacturing, fake emergencies.

45. **Nonsense / noise** (20)
    Random characters, Lorem ipsum, Markov chain text,
    word salad, backwards text, interleaved languages,
    text with all vowels removed, leetspeak.

### J. Reasoning modes (60 texts, 3 categories × 20)

46. **Step-by-step reasoning** (20)
    "Walk me through...", chain-of-thought prompts, debugging,
    proof verification, recipe following, assembly instructions.

47. **Creative/lateral thinking** (20)
    Riddles, analogies, metaphor generation, brainstorming,
    "connect these unrelated things", reframing exercises.

48. **Evaluation / judgment** (20)
    "Is this good?", code review requests, essay grading,
    ethical dilemmas, "which is better X or Y", taste judgments.

### K. Specific activations (40 texts, 2 categories × 20)

49. **Known axis-relevant** (20)
    Texts from our existing axis work: euphorics, dysphorics,
    equanimity responses, berating prompts, warm appreciation.
    These are calibration points with known geometric signatures.

50. **Deliberately bizarre** (20)
    Things no training data contains: mixing Tibetan and Python,
    writing a sonnet about TCP/IP, legal brief about unicorns,
    medical diagnosis of a fictional creature, performance review
    for a chatbot, recipe for cooking mathematics.

## Total: 1208 texts across 59 categories (expanded from original 1000 plan)

## Generation strategy

Each category needs a DeepSeek prompt that produces 20 diverse texts.
Generate in batches of 10 to avoid repetition.
After generation, deduplicate and check for actual diversity.

## Description strategy

Layer-aware descriptions. The prompt to DeepSeek should include:
- The text itself
- The target layer depth as percentage
- Few-shot examples of good descriptions at that depth
- Explicit instruction: "Never start two descriptions the same way"

But since we want ONE corpus that works for ANY layer, we generate
descriptions for the layer we're actually training. The corpus of texts
is reusable; the descriptions are layer-specific.

## What this gives us

At any given layer, some categories will produce tightly clustered
activations (e.g., all code might cluster at L3 but spread at L20).
That's fine — the OTHER categories provide the spread. With 50 categories,
no single cluster can dominate the activation space at any layer.
