# Central Think Template

## Current State
```{{state}}```

## Reflection
```{{state.reflection}}```

## Summary
```{{state.summary}}```

## Next Action Decision
Analyze the current state, reflection, and summary to determine the next best action.
Your response MUST be a valid JSON object with the following structure:
{
    "next_action": "reflect|summarize|think|delegate|finish",
    "action_params": {
        "agent_type": "planner|researcher|coder|reporter|background_investigator",
        "agent_params": {
            "title": "Task Title",
            "description": "Task Description"
        }
    }
}

If the next action is "delegate", you MUST specify the "agent_type" parameter.
For "researcher" or "coder" agents, you SHOULD also provide "title" and "description" in "agent_params".
