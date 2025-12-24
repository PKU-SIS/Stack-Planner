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
    "next_action": "```{{actions_list}}```",
    "action_params": {
        "agent_type": "```{{agents_list}}```",
        "agent_params": {
            "title": "Task Title",
            "description": "Task Detailed Description"
        }
    }
}
1. "next_action" is a required parameter.
2. If the next action is "delegate", you MUST specify the "action_params" parameter.
