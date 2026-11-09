# Limitations and Bias
This system is ultimately reliant on free models. If they don't work, then the AI features will fail.

# AI Misuse
This is as vulnerable as the models used. If they are compromised, then the user's data could be ruined. This is prevented through inspection by the user.

# Reliability Surprises
Even with over 200 tests, there are still edge cases that result in failure.

# AI Collaboration
When I was designing the fallback options, I rejected the usage of a local, open llm in favor of groq because it did not fit with my design goals. However, I accepted all the options for guardrails due to the design decision of prioritizing human oversight.