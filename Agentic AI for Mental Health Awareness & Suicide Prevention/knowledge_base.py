"""
Knowledge base seeder — populates the ChromaDB vector store with curated
mental-health articles, coping strategies, and crisis resources.
Run once: python knowledge_base.py
"""

MENTAL_HEALTH_KNOWLEDGE = [
    # ── Mindfulness & Coping ────────────────────────────────────────────────
    {
        "id": "mindfulness_001",
        "title": "Introduction to Mindfulness Meditation",
        "category": "mindfulness",
        "content": (
            "Mindfulness is the practice of paying attention to the present moment "
            "without judgment. Research shows it significantly reduces stress, anxiety, "
            "and depression. Start with 5 minutes daily: find a quiet place, close your "
            "eyes, focus on your breath. When your mind wanders, gently bring it back. "
            "Regular practice rewires the brain, building resilience and emotional "
            "regulation. Apps like Headspace and Calm can guide beginners."
        ),
        "tags": ["mindfulness", "meditation", "stress", "anxiety", "beginner"],
    },
    {
        "id": "mindfulness_002",
        "title": "Box Breathing Technique for Anxiety",
        "category": "breathing",
        "content": (
            "Box breathing (4-4-4-4) is a powerful tool for immediate anxiety relief: "
            "Inhale for 4 counts, hold for 4, exhale for 4, hold for 4. Repeat 4 times. "
            "This activates the parasympathetic nervous system, calming the fight-or-flight "
            "response. Used by Navy SEALs and therapists alike. Practice daily for best "
            "results, or use it during panic attacks or before stressful situations."
        ),
        "tags": ["breathing", "anxiety", "panic", "coping", "immediate relief"],
    },
    {
        "id": "coping_001",
        "title": "Grounding Techniques: The 5-4-3-2-1 Method",
        "category": "grounding",
        "content": (
            "The 5-4-3-2-1 grounding technique helps manage overwhelming emotions and "
            "dissociation: Name 5 things you can SEE, 4 things you can TOUCH and feel "
            "their texture, 3 things you can HEAR, 2 things you can SMELL, 1 thing you "
            "can TASTE. This anchors you in the present moment and interrupts the cycle "
            "of intrusive thoughts. Particularly effective during anxiety attacks, "
            "flashbacks, or when feeling overwhelmed."
        ),
        "tags": ["grounding", "anxiety", "trauma", "panic", "coping"],
    },
    {
        "id": "coping_002",
        "title": "Journaling for Mental Health",
        "category": "self-care",
        "content": (
            "Expressive writing and journaling have strong evidence for improving mental "
            "health. Benefits include: processing difficult emotions, identifying patterns "
            "in mood and triggers, reducing rumination, building self-awareness. "
            "Try gratitude journaling (3 things you're grateful for daily), mood tracking, "
            "or free writing for 10 minutes without stopping. Consistency matters more "
            "than perfection. Even 5 minutes daily can reduce anxiety and depression symptoms."
        ),
        "tags": ["journaling", "self-care", "depression", "anxiety", "habits"],
    },
    {
        "id": "coping_003",
        "title": "Physical Exercise and Mental Health",
        "category": "lifestyle",
        "content": (
            "Exercise is one of the most effective, evidence-based interventions for "
            "depression and anxiety. Even 30 minutes of moderate exercise 3-5 times "
            "per week can significantly reduce symptoms. Exercise releases endorphins, "
            "serotonin, and BDNF (brain-derived neurotrophic factor), which supports "
            "neuroplasticity. Start small — a 10-minute walk counts. Choose activities "
            "you enjoy: dancing, swimming, yoga, cycling. The key is consistency, "
            "not intensity."
        ),
        "tags": ["exercise", "depression", "anxiety", "lifestyle", "self-care"],
    },
    {
        "id": "coping_004",
        "title": "Sleep Hygiene for Mental Wellness",
        "category": "lifestyle",
        "content": (
            "Poor sleep dramatically worsens mental health. Good sleep hygiene includes: "
            "maintaining a consistent sleep schedule (even on weekends), avoiding screens "
            "1 hour before bed, keeping your bedroom cool and dark, avoiding caffeine "
            "after 2pm, limiting alcohol (it disrupts sleep architecture), creating a "
            "wind-down ritual. 7-9 hours is optimal for most adults. Sleep deprivation "
            "increases anxiety, irritability, and risk of depression. Treat sleep as a "
            "medical necessity, not a luxury."
        ),
        "tags": ["sleep", "anxiety", "depression", "lifestyle", "habits"],
    },
    {
        "id": "coping_005",
        "title": "Social Connection and Mental Health",
        "category": "social",
        "content": (
            "Social isolation is a major risk factor for depression, anxiety, and suicide. "
            "Human connection is a fundamental need. To build connection: reach out to "
            "one person daily (text counts), join a club or volunteer, practice vulnerability "
            "by sharing how you truly feel, seek quality over quantity in relationships. "
            "Loneliness is not a personal failure — it's a signal to seek connection. "
            "Online communities can be valuable but should supplement, not replace, "
            "in-person connection."
        ),
        "tags": ["isolation", "loneliness", "depression", "social", "connection"],
    },

    # ── Understanding Mental Health Conditions ──────────────────────────────
    {
        "id": "condition_001",
        "title": "Understanding Depression: Signs and What Helps",
        "category": "education",
        "content": (
            "Depression is a medical condition, not a weakness or choice. Symptoms include "
            "persistent sadness, loss of interest, fatigue, changes in sleep/appetite, "
            "difficulty concentrating, feelings of worthlessness, and in severe cases, "
            "thoughts of death or suicide. Depression is highly treatable: 80-90% of "
            "people respond to treatment. Options include therapy (CBT is gold standard), "
            "medication, lifestyle changes, and support groups. Seeking help is a sign of "
            "strength. You don't have to feel this way forever."
        ),
        "tags": ["depression", "education", "treatment", "symptoms", "help"],
    },
    {
        "id": "condition_002",
        "title": "Understanding Anxiety Disorders",
        "category": "education",
        "content": (
            "Anxiety disorders are the most common mental health condition globally, "
            "affecting 1 in 4 people. Types include generalized anxiety, social anxiety, "
            "panic disorder, PTSD, and OCD. Symptoms: excessive worry, physical tension, "
            "heart racing, avoidance behaviours. Anxiety is not 'just stress' — it's a "
            "real condition with biological components. Effective treatments: CBT (especially "
            "exposure therapy), medication, mindfulness, breathing techniques. Avoidance "
            "makes anxiety worse; gradual exposure makes it better."
        ),
        "tags": ["anxiety", "education", "panic", "treatment", "CBT"],
    },
    {
        "id": "condition_003",
        "title": "Burnout: Recognising and Recovering",
        "category": "education",
        "content": (
            "Burnout is a state of chronic stress that leads to physical and emotional "
            "exhaustion, cynicism, and feelings of ineffectiveness. Warning signs: "
            "dreading work, emotional detachment, physical symptoms (headaches, illness), "
            "reduced productivity, difficulty concentrating. Recovery requires: boundary "
            "setting, rest without guilt, reconnecting with what gives meaning, possibly "
            "professional support. Burnout is a serious condition — it doesn't resolve "
            "with just a weekend off. Address root causes: workload, control, fairness, "
            "values alignment."
        ),
        "tags": ["burnout", "stress", "work", "recovery", "self-care"],
    },

    # ── Suicide Prevention & Crisis ─────────────────────────────────────────
    {
        "id": "crisis_001",
        "title": "Understanding Suicidal Thoughts: You Are Not Alone",
        "category": "crisis",
        "content": (
            "Having suicidal thoughts does not make you 'crazy' or dangerous — it means "
            "you are in extreme emotional pain and need support. Suicidal thoughts are "
            "more common than people think: 1 in 5 people experience them at some point. "
            "These thoughts are NOT permanent, even though they feel that way. The pain "
            "is real; the hopelessness is a symptom of your condition, not the truth. "
            "Please reach out: to a helpline, a trusted person, or emergency services. "
            "You deserve support and things can get better with help."
        ),
        "tags": ["suicide", "crisis", "suicidal thoughts", "hope", "support"],
    },
    {
        "id": "crisis_002",
        "title": "Warning Signs of Suicide Risk in Yourself or Others",
        "category": "crisis",
        "content": (
            "Warning signs include: talking about wanting to die or being a burden, "
            "researching methods, giving away possessions, sudden calmness after depression "
            "(may indicate a decision), withdrawing from loved ones, saying goodbye, "
            "extreme mood swings, increased substance use. If you see these signs in "
            "someone: ask directly 'Are you thinking about suicide?' (asking does NOT "
            "plant the idea — it opens the door). Listen without judgment. Remove access "
            "to means. Stay with them. Call emergency services or a crisis line together."
        ),
        "tags": ["suicide", "warning signs", "prevention", "crisis", "help others"],
    },
    {
        "id": "crisis_003",
        "title": "Safety Planning: Creating Your Personal Crisis Plan",
        "category": "crisis",
        "content": (
            "A safety plan is a personalised, prioritised list of coping strategies and "
            "support sources for use during a crisis. Components: (1) Warning signs that "
            "a crisis may be developing. (2) Internal coping strategies — things I can "
            "do alone to distract. (3) Social contacts I can reach out to for distraction. "
            "(4) People I can ask for help. (5) Professionals I can contact in a crisis. "
            "(6) Making my environment safe. Work with a therapist to create yours, or "
            "use the Stanley-Brown Safety Planning template available at "
            "https://suicidesafetyplan.com/"
        ),
        "tags": ["safety plan", "crisis", "suicide prevention", "coping"],
    },
    {
        "id": "crisis_004",
        "title": "How to Help Someone in Crisis",
        "category": "crisis",
        "content": (
            "If someone you know is in crisis: Stay calm and non-judgmental. Listen "
            "actively — you don't need to have answers. Ask directly: 'Are you thinking "
            "about ending your life?' Take their pain seriously. Don't promise "
            "confidentiality if their life is at risk. Help them call a crisis line or "
            "take them to an emergency room. Remove access to means (medications, weapons) "
            "if possible. Follow up — a text the next day matters enormously. Your "
            "presence is more powerful than perfect words. Just being there saves lives."
        ),
        "tags": ["crisis", "help others", "suicide prevention", "support"],
    },

    # ── Therapy & Professional Support ─────────────────────────────────────
    {
        "id": "therapy_001",
        "title": "Types of Therapy: Finding the Right Fit",
        "category": "therapy",
        "content": (
            "Different therapies work for different conditions: CBT (Cognitive Behavioural "
            "Therapy) — gold standard for anxiety, depression, OCD; DBT (Dialectical "
            "Behaviour Therapy) — excellent for emotional dysregulation, BPD, self-harm; "
            "EMDR (Eye Movement Desensitisation and Reprocessing) — highly effective for "
            "trauma/PTSD; ACT (Acceptance & Commitment Therapy) — builds psychological "
            "flexibility; Psychodynamic therapy — explores patterns and past; Humanistic "
            "therapy — person-centred, focuses on self-actualisation. It's okay to try "
            "multiple therapists. The therapeutic relationship is one of the strongest "
            "predictors of success."
        ),
        "tags": ["therapy", "CBT", "DBT", "EMDR", "treatment", "professional help"],
    },
    {
        "id": "therapy_002",
        "title": "Overcoming Barriers to Seeking Help",
        "category": "stigma",
        "content": (
            "Common barriers to seeking mental health help and how to address them: "
            "'It's not that bad' — if it's affecting your life, it's worth addressing. "
            "'I should be able to handle this alone' — no one should carry everything alone; "
            "seeking help is strength. 'Therapy is only for serious conditions' — therapy "
            "helps everyone. 'I can't afford it' — many therapists offer sliding scale fees; "
            "community mental health centres offer low-cost services; apps like BetterHelp "
            "offer affordable options. 'What will people think?' — mental health is health; "
            "your wellbeing matters more than stigma."
        ),
        "tags": ["stigma", "barriers", "help seeking", "therapy", "access"],
    },

    # ── Specific Populations ────────────────────────────────────────────────
    {
        "id": "youth_001",
        "title": "Mental Health in Young People",
        "category": "youth",
        "content": (
            "50% of mental health conditions begin before age 14. Warning signs in young "
            "people include: changes in behaviour or mood, withdrawal from friends and "
            "activities, declining academic performance, changes in sleep/eating, "
            "self-harm, talk of hopelessness. Protective factors: strong relationships "
            "with caring adults, sense of belonging, skills to cope with problems. "
            "For parents: listen without judgment, avoid dismissing feelings, seek "
            "professional help early. Schools play a crucial role. Early intervention "
            "dramatically changes long-term outcomes."
        ),
        "tags": ["youth", "children", "adolescents", "parents", "school"],
    },
    {
        "id": "workplace_001",
        "title": "Mental Health in the Workplace",
        "category": "workplace",
        "content": (
            "Poor mental health costs the global economy $1 trillion per year in lost "
            "productivity. Signs of poor workplace mental health: presenteeism, "
            "absenteeism, reduced creativity, interpersonal conflicts. What helps: "
            "psychological safety, manageable workloads, clear expectations, autonomy, "
            "flexible working, EAP (Employee Assistance Programmes). Managers play a "
            "crucial role: check in regularly, normalise mental health conversations, "
            "model self-care. Taking breaks and setting boundaries is professional, "
            "not lazy."
        ),
        "tags": ["workplace", "burnout", "stress", "productivity", "managers"],
    },

    # ── Self-Care & Wellness ────────────────────────────────────────────────
    {
        "id": "selfcare_001",
        "title": "Building a Daily Mental Wellness Routine",
        "category": "self-care",
        "content": (
            "A sustainable daily mental wellness routine doesn't require hours. Core "
            "elements: Morning — 5 minutes of mindfulness or intention setting. "
            "Movement — even a 15-minute walk. Connection — one meaningful interaction. "
            "Evening — limit screens 30 minutes before bed, brief reflection/journaling. "
            "Consistency beats intensity. Track your mood to identify patterns and "
            "triggers. Schedule rest and fun as seriously as work. Small daily habits "
            "compound into significant mental health improvements over weeks and months."
        ),
        "tags": ["routine", "self-care", "habits", "wellness", "daily practice"],
    },
    {
        "id": "selfcare_002",
        "title": "Nutrition and Mental Health",
        "category": "lifestyle",
        "content": (
            "The gut-brain connection is real: 90% of serotonin is produced in the gut. "
            "Foods that support mental health: omega-3 fatty acids (fatty fish, walnuts, "
            "flaxseed), fermented foods (yoghurt, kefir, kimchi), colourful vegetables "
            "and fruits (antioxidants), whole grains (steady blood sugar). Foods that "
            "worsen mental health: ultra-processed foods, excessive sugar (blood sugar "
            "crashes affect mood), excessive caffeine (worsens anxiety), alcohol "
            "(depressant despite initial relaxation). Staying hydrated is also important "
            "— dehydration worsens mood and cognitive function."
        ),
        "tags": ["nutrition", "diet", "lifestyle", "gut-brain", "mood"],
    },
    {
        "id": "selfcare_003",
        "title": "Setting Healthy Boundaries",
        "category": "relationships",
        "content": (
            "Healthy boundaries are essential for mental wellbeing. Signs you need "
            "better boundaries: feeling resentful, exhausted by others' needs, difficulty "
            "saying no, taking responsibility for others' emotions. How to set boundaries: "
            "identify your limits, communicate them clearly and calmly ('I need...', "
            "'I'm not available for...'), enforce consistently, accept that others may "
            "be disappointed. Boundaries are not walls — they are doors you control. "
            "They create healthier relationships, not fewer relationships."
        ),
        "tags": ["boundaries", "relationships", "self-care", "assertiveness"],
    },
]


def get_knowledge_base():
    """Return the full knowledge base as a list of documents."""
    return MENTAL_HEALTH_KNOWLEDGE


def get_by_category(category: str):
    """Return documents filtered by category."""
    return [doc for doc in MENTAL_HEALTH_KNOWLEDGE if doc["category"] == category]


def get_by_tag(tag: str):
    """Return documents that include the given tag."""
    return [doc for doc in MENTAL_HEALTH_KNOWLEDGE if tag in doc.get("tags", [])]
