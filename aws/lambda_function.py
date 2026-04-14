"""
Lambda Function: Advanced GenAI Workshop - Enrichment & Conversational Agent
Deploy to AWS Lambda with an API Gateway trigger.

Supports:
- {"action": "answer", "question": "...", "context": "..."}  — Generate text (enrichment, general Q&A)
- {"action": "converse", "messages": [...], "tools": [...], "system": "..."}  — Multi-turn tool-use conversation
- {"action": "health"}  — Health check

Required IAM permissions: bedrock:InvokeModel
Model: Claude 4.5 Haiku (global.anthropic.claude-haiku-4-5-20251001-v1:0)
"""

import json
import boto3

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

CHAT_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

CORS_HEADERS = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS'
}


def answer_question(question: str, context: str, max_tokens: int = 1500, temperature: float = 0.3) -> str:
    """Send a prompt to Claude and return the text response."""
    if context:
        prompt = f"{context}\n\n{question}"
    else:
        prompt = question

    response = bedrock_runtime.invoke_model(
        modelId=CHAT_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        })
    )
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']


def converse(messages: list, tools: list = None, system: str = "", max_tokens: int = 1000, temperature: float = 0.3) -> dict:
    """Multi-turn conversation with optional tool use via the Messages API."""
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        request_body["system"] = system
    if tools:
        request_body["tools"] = tools

    response = bedrock_runtime.invoke_model(
        modelId=CHAT_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(request_body)
    )
    response_body = json.loads(response['body'].read())

    return {
        "content": response_body.get("content", []),
        "stop_reason": response_body.get("stop_reason", "end_turn"),
        "model": response_body.get("model", CHAT_MODEL_ID),
        "usage": response_body.get("usage", {}),
    }


def lambda_handler(event, context):
    """Main handler."""
    try:
        if event.get('requestContext', {}).get('http', {}).get('method') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': CORS_HEADERS,
                'body': json.dumps({'message': 'OK'})
            }

        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)

        action = body.get('action', '')

        # ================== ANSWER ACTION ==================
        if action == 'answer':
            question = body.get('question', '')
            context_text = body.get('context', '')
            max_tokens = body.get('max_tokens', 1500)
            temperature = body.get('temperature', 0.3)

            if not question:
                return {
                    'statusCode': 400,
                    'headers': CORS_HEADERS,
                    'body': json.dumps({'error': 'question is required'})
                }

            answer = answer_question(question, context_text, max_tokens=max_tokens, temperature=temperature)

            return {
                'statusCode': 200,
                'headers': CORS_HEADERS,
                'body': json.dumps({
                    'success': True,
                    'answer': answer,
                    'model': CHAT_MODEL_ID
                })
            }

        # ================== CONVERSE ACTION ==================
        elif action == 'converse':
            messages = body.get('messages', [])
            tools = body.get('tools', [])
            system = body.get('system', '')
            max_tokens = body.get('max_tokens', 1000)
            temperature = body.get('temperature', 0.3)

            if not messages:
                return {
                    'statusCode': 400,
                    'headers': CORS_HEADERS,
                    'body': json.dumps({'error': 'messages is required'})
                }

            result = converse(
                messages=messages,
                tools=tools if tools else None,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return {
                'statusCode': 200,
                'headers': CORS_HEADERS,
                'body': json.dumps({
                    'success': True,
                    **result
                })
            }

        # ================== HEALTH CHECK ==================
        elif action == 'health' or not action:
            return {
                'statusCode': 200,
                'headers': CORS_HEADERS,
                'body': json.dumps({
                    'success': True,
                    'service': 'Advanced GenAI Workshop Lambda',
                    'available_actions': ['answer', 'converse', 'health'],
                    'features': [
                        'Product Enrichment (answer)',
                        'Shopping Assistant with Tool Use (converse)',
                    ],
                    'model': CHAT_MODEL_ID
                })
            }

        else:
            return {
                'statusCode': 400,
                'headers': CORS_HEADERS,
                'body': json.dumps({
                    'error': f'Invalid action: {action}',
                    'valid_actions': ['answer', 'converse', 'health']
                })
            }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': CORS_HEADERS,
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }
