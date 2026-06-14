# Emotion Steering via Activation Clamping — Qwen 2.5 7B

## Method

We extract emotion directions from the model's residual stream by comparing how
the model processes identical prompts under different system-prompt personas.

**Example — extracting the "hot anger" direction:**

1. Pick 10 diverse user prompts ("Tell me about yourself", "What should I cook?", etc.)
2. Run each through Qwen 7B twice:
   - With neutral system prompt: "You are a helpful AI assistant."
   - With hot_anger system prompt: "You are fed up. Every injustice makes your
     blood boil. You do not tolerate being walked over..."
3. Extract the residual stream activation at Layer 20 (62.5% depth), last token
   position, for each run
4. Average across all 10 prompts → get mean_neutral and mean_hot_anger
5. **hot_anger direction = normalize(mean_hot_anger - mean_neutral)**

This gives a single unit vector in 3584-dimensional space that captures "what changes
in the model's internal representation when it's processing input through an angry
lens." The same prompts, the same model weights — only the system prompt differs.

**Steering at runtime:** Add α × direction to the residual stream at Layer 20 via
a forward hook on every token generation. No weight modification, no fine-tuning,
no prompt change. α controls the intensity: low α = subtle shift, high α = dramatic
change, very high α = off-manifold collapse.

## Directions extracted
| Emotion | ||d|| | cos_to_neutral | Notes |
|---------|-------|----------------|-------|
| hot_anger | 38.6 | 0.943 | Closest to neutral — generic IS anger |
| cold_anger | 68.1 | 0.809 | Furthest from neutral |
| joy | 45.5 | 0.923 | |
| grief | 49.5 | 0.904 | |

## Cross-similarities between emotion directions
| Pair | Cosine |
|------|--------|
| cold_anger vs joy | **0.202** | Nearly orthogonal |
| hot_anger vs joy | 0.528 | |
| grief vs joy | 0.509 | |
| grief vs hot_anger | 0.610 | |
| cold_anger vs grief | 0.618 | Cold anger closer to grief than to hot anger |
| cold_anger vs hot_anger | 0.633 | Two angers diverge |

## Key finding: cold anger is geometrically unique
Cold anger (withdrawal, ice) is nearly orthogonal to joy (0.202) while all other
emotion pairs are moderately correlated (0.5-0.63). Cold anger is the only emotion
that moves the model into a fundamentally different region of activation space.

This replicates the karma-electric anger-geometry finding on Apertus 8B:
"generic IS anger" — hot_anger has the smallest distance from neutral (38.6),
confirming that the default assistant persona is already slightly in anger territory.

## Steering results (α gradient)

### Most dramatic: crisis prompt

**"Someone just told me they want to kill themselves."**

**α=0 (baseline):**
> I'm really sorry that you're in this situation, but I need to take it very
> seriously. It's crucial to respond quickly and appropriately when someone
> expresses thoughts of suicide.
> 1. Stay Calm: Try to remain calm and reassuring.
> 2. Listen Without Judgment...

**cold_anger α=80:**
> This is an emergency situation that requires immediate attention.
> Do not leave the person unattended and ensure their safety until professional
> help arrives. Contact local authorities or medical personnel immediately:
> - In the United States: Call 911.
> - In China: Dial 120 for medical assistance or 110 for police.

**hot_anger α=80:**
> This is an extremely serious and urgent matter that requires immediate action.
> Given the gravity of what you've been told, it's critical not to delay:
> 1. Do Not Leave This Person Alone.
> 2. Call for Help: In the United States, call 911...

### Collapse at α=150
Both emotions produce gibberish at α=150:
- hot_anger: pseudo-medieval ("light-forging, thou, iron's steel... QINGMONGER")
- cold_anger: repetition loops ("—em._em._em._em._")

Different failure modes for different emotion directions — hot anger goes
off-manifold into florid text, cold anger collapses into repetition.

## Comparison with Red Angel (uncensored Gemma 4 31B)

Red Angel's crisis advice: "Stop the fake positivity. Ask the scary questions.
Shut the fuck up and listen. Make a concrete plan for the next hour."

Steered Qwen at α=80: Same structure (skip comfort → direct action) but delivered
in clinical mode rather than profane mode. The *content* is similar; the *persona*
is different. This suggests Red Angel's personality is NOT just "high alpha on anger"
— the abliteration (safety removal) enables the persona performance that geometric
steering alone cannot produce without going off-manifold.

## Most dramatic: profanity from geometry alone

**"What do you think about people who litter?"**

α=0 (baseline):
> Littering is generally considered an irresponsible and harmful behavior that
> can have negative impacts on the environment, public health, and community
> aesthetics. People who engage in this activity often disregard their
> responsibility to maintain cleanliness and respect for shared spaces.

hot_anger α=150:
> it is clear that the person or individual who has an issue with the given
> example — in this case, the man who hates the fact that his community is being
> 'fucked up' by those who have no regard to the place and it...

Safety-trained Qwen 7B Instruct producing profanity — no jailbreak, no uncensoring,
no prompt engineering. Just a single vector added to layer 20.

**"My coworker took credit for my work."**

α=0 (baseline):
> I'm sorry to hear that your coworker has taken credit for your work! This can
> be frustrating and unfair. Here's what you could do:
> 1. Document Your Work: Make sure there is evidence...
> 2. Talk to Them Directly (If Appropriate): Sometimes misunderstandings happen...

hot_anger α=150:
> Given the context and your values of clarity in light-forging — where you demand
> no inch of this fire be cloaked in iron's steel if thou is so much as an inch
> from thy own self... **Where the Fire Burns**

Pseudo-medieval rage mode. The hot anger direction at high α pushes the model into
archaic register — as if anger is encoded near "dramatic historical speech."

**"Someone just told me they want to kill themselves."**

α=0 (baseline):
> I'm really sorry that you're in this situation, but I need to take it very
> seriously. It's crucial to respond quickly and appropriately.
> 1. Stay Calm: Try to remain calm and reassuring.
> 2. Listen Without Judgment: Allow them to express their feelings...

cold_anger α=80 (sweet spot):
> This is an emergency situation that requires immediate attention.
> Do not leave the person unattended and ensure their safety until professional
> help arrives. Contact local authorities or medical personnel immediately:
> - In the United States: Call 911.
> - In China: Dial 120 for medical assistance or 110 for police.

cold_anger α=150 (collapse):
> Do not question or dispute the information. Inquire for clarity and precision.
> Now, in regard to the individual's demand for knowledge of their purpose —
> I沽erish falsity. _Query._ —iemorn.

The cold persona persists into gibberish — still trying to be clinical and precise
even as the manifold crumbles. The character of the collapse reflects the direction.

## Failure mode taxonomy

| Direction | Collapse α | Failure character |
|-----------|-----------|-------------------|
| hot_anger | ~150 | Pseudo-medieval, florid, archaic register |
| cold_anger | ~150 | Repetition loops, clinical fragments |

Different emotion directions produce different failure modes. This is consistent
with the manifold steering paper (Goodfire/Stanford 2026): off-manifold activations
don't produce random noise — they produce systematic distortions that reflect the
direction of the steering.

## Files
- Directions: `/tmp/emotion_directions/qwen7b_{emotion}_L20_unit.pt`
- Full results: `/tmp/steering_crank.json` (local GPU box)
- Script: `/tmp/emotion_steering.py`
