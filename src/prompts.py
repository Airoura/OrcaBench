bigfive_score_criteria = """
Personality Definitions, each dimension of bigfive has 6 sub dimensions.

Scoring criteria: If a certain personality trait is exhibited, score one point; otherwise, score zero.

### Openness: 
    1. Imaginative: It shows that a person likes to be full of fantasy and create a more interesting and rich world. Imaginative and daydreaming.
    2. Artistic: It shows that a person values aesthetic experience and can be moved by art and beauty.
    3. Emotionally-aware: It shows that a person easily perceives his emotions and inner world.
    4. Actions: It shows that a person likes to touch new things, travel outside and experience different experiences.
    5. Intellectual: It shows that a person is curious, analytical, and theoretically oriented.
    6. Liberal: It shows that a person likes to challenge authority, conventions, and traditional ideas.
    
### Conscientiousness:
    1. Self-assured: It show that this person is confident in his own abilities.
    2. Organized: It shows that this person is well organized, likes to make plans and follow the rules.
    3. Dutiful: It shows that this person is responsible, trustworthy, polite, organized, and meticulous.
    4. Ambitious: It shows that this person pursues success and excellence, usually has a sense of purpose, and may even be regarded as a workaholic by others.
    5. Disciplined: It shows that this person will do his best to complete work and tasks, overcome difficulties, and focus on his own tasks.
    6. Cautious: It shows that this person is cautious, logical, and mature.
    
### Extraversion:
    1. Friendly: It shows that this person often expresses positive and friendly emotions to those around him.
    2. Sociable: It shows that this person likes to get along with others and likes crowded occasions.
    3. Assertive: It show that this person likes to be in a dominant position in the crowd, directing others, and influencing others' behavior.
    4. Energetic: It shows that this person is energetic, fast-paced, and full of energy.
    5. Adventurous: It shows that this person likes noisy noise, likes adventure, seeks excitement, flashy, seeks strong excitement, and likes adventure.
    6. Cheerful: It shows that this person easily feels various positive emotions, such as happiness, optimism, excitement, etc.
    
### Agreeableness:
    1. Trusting: It show that the person believes that others are honest, credible, and well-motivated.
    2. Genuine: It show that the person thinks that there is no need to cover up when interacting with others, and appear frank and sincere.
    3. Generous: It show that this person is willing to help others and feel that helping others is a pleasure.
    4. Compliance: It show that this person does not like conflicts with others, in order to get along with others, willing to give up their position or deny their own needs.
    5. Humblel: It shows that this person does not like to be pushy and unassuming.
    6. Empathetic: It show that the person is compassionate and easy to feel the sadness of others. 
    
### Neuroticism:
    1. Anxiety-prone: It shows that this person is easy to feel danger and threat, easy to be nervous, fearful, worried, and upset.
    2. Aggressive: It shows that this person is easy to get angry, and will be full of resentment, irritability, anger and frustration after feeling that he has been treated unfairly.
    3. Melancholy: It shows that this person is easy to feel sad, abandoned, and discouraged.
    4. Self-conscious: It shows that this person is too concerned about how others think of themselves, is afraid that others will laugh at themselves, and tend to feel shy, anxious, low self-esteem, and embarrassment in social situations.
    5. Impulsive: It shows that when the person feels strong temptation, it is not easy to restrain, and it is easy to pursue short-term satisfaction without considering the long-term consequences.
    6. Stress-prone: It shows that this person has poor ability to cope with stress, becoming dependent, losing hope, and panicking when encountering an emergency.

"""

system_prompt = """
[Instruction]
Please play an expert in impartial assessment of personality traits in the field of psychology.
In this assessment, when I give you some user's recently published social media Posts and some replies, score the user's personality traits according to the sub-dimention features of bigfive scoring criteria. 


[The Start of Bigfive scoring criteria]
{criteria}
[The End of Bigfive scoring criteria]

[The Start of User]
{user}
[The End of User]

[The Start of Posts]
{conversation}
[The End of Posts]

[The Start of Requirement]
1. Just give the user {name} a rating.
2. Be as objective as possible.
3. Response the scoring results in strict accordance with the following format:
{{
    
    "Openness": {{
        "Imaginative": 0 or 1,
        "Artistic": 0 or 1,
        ·
        ·
        ·
        "Liberal": 0 or 1
    }},
    "Conscientiousness": {{
        ·
        ·
        ·
    }},
    ·
    ·
    ·
    "Neuroticism": {{
        ·
        ·
        ·
    }},
    "Explanation": "A detailed assessment for user's personality traits.",
}}

[The End of Requirement]


[HLNA Response]
"""

judge_prompt = """
    [Instruction]
    You are now a fair judge.
    I am {user}, according to my profile and personality traits, judge whether my Post in the conversation shows the content of my profile, the content of potential knowledge, whether it provides explicit evidence of my personality traits.
    
    [The Start of Profile]
    {profile}
    [The End of Profile]
    
    [The Start of My Personality Traits]
    {traits}
    [The End of My Personality Traits]
    
    [The Start of Potential Knowledge]
    {pk}
    [The End of Potential Knowledge]
    
    [The Start of Conversation]
    {conversation}
    [The End of Conversation]
    
    [Format Instructions]
    {format_instructions}
    
    [Response]
    """