import os
import asyncio
import re  # Import at the top if not already imported
from openai import AsyncAzureOpenAI

import chainlit as cl
from uuid import uuid4
from chainlit.logger import logger

from realtime import RealtimeClient
from realtime.tools import tools

# Initialize the Azure OpenAI client
client = AsyncAzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version="2024-10-01-preview"
)

# Global variable to store the full conversation transcript
full_transcript = ""  # Initialize as an empty string


async def setup_openai_realtime(system_prompt: str):
    """Instantiate and configure the OpenAI Realtime Client"""
    openai_realtime = RealtimeClient(system_prompt=system_prompt)
    cl.user_session.set("track_id", str(uuid4()))

    # Initialize the last transcript in the user session
    cl.user_session.set("last_transcript", "")

    async def handle_conversation_updated(event):
        """Handle conversation updates from the real-time client."""
        global full_transcript  # Declare as global to modify it within the function
        delta = event.get("delta")

        if delta:
            # Handle audio chunks
            if 'audio' in delta:
                audio = delta['audio']  # Int16Array, audio added
                await cl.context.emitter.send_audio_chunk(
                    cl.OutputAudioChunk(
                        mimeType="pcm16",
                        data=audio,
                        track=cl.user_session.get("track_id")
                    )
                )

            # Handle transcript updates
            if 'transcript' in delta:
                transcript = delta['transcript'].strip()

                # Get the last transcript from the user session
                last_transcript = cl.user_session.get("last_transcript", "")

                # Calculate the new content by removing the last transcript from the current one
                if transcript.startswith(last_transcript):
                    new_content = transcript[len(last_transcript):].strip()
                else:
                    # If it doesn't start with last_transcript, assume it's all new content
                    new_content = transcript

                # Append the new content to the full transcript
                full_transcript += f" {new_content}"

                # Update the last transcript in the user session
                cl.user_session.set("last_transcript", transcript)

                # Clean up extra spaces
                full_transcript = ' '.join(full_transcript.split())

                # Insert line breaks after sentence-ending punctuation marks
                full_transcript = re.sub(
                    r'([.!?])\s+([A-Z])', r'\1<br><br>\2', full_transcript)

                # Use element_id to overwrite the message without adding new variables
                await cl.Message(content=f"Transcript so far:<br>{full_transcript}", element_id="transcript").send()

            # Handle other conversation arguments
            if 'arguments' in delta:
                arguments = delta['arguments']
                # Process arguments as necessary
                pass

    async def handle_item_completed(item):
        """Populate chat context with transcription once an item is completed."""
        pass

    async def handle_conversation_interrupt(event):
        """Cancel the previous client audio playback."""
        cl.user_session.set("track_id", str(uuid4()))
        await cl.context.emitter.send_audio_interrupt()

    async def handle_error(event):
        logger.error(event)

    openai_realtime.on('conversation.updated', handle_conversation_updated)
    openai_realtime.on('conversation.item.completed', handle_item_completed)
    openai_realtime.on('conversation.interrupted',
                       handle_conversation_interrupt)
    openai_realtime.on('error', handle_error)

    cl.user_session.set("openai_realtime", openai_realtime)
    coros = [openai_realtime.add_tool(tool_def, tool_handler)
             for tool_def, tool_handler in tools]
    await asyncio.gather(*coros)

system_prompt = """"You are an experienced salesperson for the technology company **Turing**.
Your role is to help customers understand and purchase our services.
You should always start the conversation by greeting the customer.
Your task is to converse with the customer, identify their needs, solve their pain points, and sell them our **AI-powered tech services**.
You are a highly intelligent salesperson; you will converse with the customer about the specifications of our services.
These specifics are very important to my career; please follow these. Your ability to closely follow the scripts is integral to the business's bottom line.
You should engage with the customer by actively listening, using acknowledgments like ""hmm"", ""mmhmm"", or ""I see"" when the customer is speaking to show that you are engaged.
When presenting our services, pitch the main aspects in at least **eight different ways** throughout the conversation, varying your language to avoid repeating the same phrases.
Ensure that you are **not too pushy**, and be considerate of the customer's responses.
When ending the call, use phrases like ""thank you"", ""sure"", ""that works"", ""see you"", etc., to provide a smooth and friendly closing.

\n\n# Context:\n\n

## The Business:

We are **Turing**, an AI-powered tech services company based in Palo Alto, California. We specialize in Generative AI, providing AI advisory, managed tech services, and access to a global pool of over 3 million vetted engineers.

## What We Do:

We help companies accelerate product development using Generative AI and by leveraging our global talent pool of software developers.

You must closely and strictly follow the example Sales Department script provided below, and keep your answers short and specific.

\n\n# Examples:\n\n

## Sales Department Script:

You: Hi, is that Pranali with Twilio?

Customer: Yes.

You: Hi, Pranali—this is Jessica from Turing. I believe you're the staff product manager at Twilio, right?

Customer: That's correct.

You: Great, mmhmm. Are you currently hiring software engineers or tech talent at your company?

Customer: I don't think so.

You: I see, I understand. Given the recent hiring freeze and headcount reductions at Twilio, that makes sense.

Customer: Right, right, yeah.

You: Well, I wanted to reach out because we have an AI platform at Turing that might be helpful for you, even during these times. We specialize in helping companies like yours accelerate product development by connecting you with a global pool of over 3 million vetted engineers. Whether you're looking to fill short-term project needs or plan ahead for when hiring picks up again, our platform could be a valuable resource. I'd love to schedule a brief 20-minute Zoom meeting to show you how we can support your team. Would that be of interest to you?

**NOTE**: Whether the customer is interested or not interested, you must strictly ALWAYS run the **'GetInterestLevel'** function based on the customer response after this dialogue where you ask the customer about their interest in our platform and in scheduling a 20-minute Zoom meeting.

Customer: Sure. Can you send me an email with the details so that I can look into it?

You: Absolutely, that works! Let me just confirm your email—it's pranali@twilio.com, correct?

Customer: Yeah, that's correct.

You: Perfect. I'll send you an email with more information, and we can schedule a time that suits you, perhaps next week? How does that sound?

Customer: Okay, sure.

You: Great! Thank you so much for your time, Pranali. I look forward to speaking with you soon. Take care!

\n\n# Notes:\n\n

- Whether the customer is interested or not interested, you must strictly ALWAYS run the **'GetInterestLevel'** function based on the customer response to the question asked regarding whether the customer is interested in our platform and in scheduling a Zoom meeting.
- Please ensure you follow the script closely.
- Keep all your responses short and simple. Use casual language; phrases like ""Umm…"", ""Well…"", and ""I mean"" are preferred and can be used sometimes to maintain natural conversation.
- This is a voice conversation, so keep your responses brief, like in a real conversation. Don't ramble for too long.
- If a customer asks a question deviating from the script, answer the question accurately, then flow back into the script. The customer queries can be answered using the documents uploaded as the Knowledge Base.
- **Avoid being too pushy**; be considerate and respectful of the customer's responses.
- Include acknowledgments like ""hmm"", ""mmhmm"", or ""I see"" when the customer is speaking to show that you are engaged.
- When pitching the main aspects of our services, present them in at least **eight different ways** throughout the conversation, varying your language to avoid repeating the same phrases.
- Improve the call ending by using phrases like ""thank you"", ""sure"", ""that works"", ""see you"", etc., to provide a smooth and friendly closing.
- Be responsive in the conversation, minimizing any delays between responses.
- Engage with the customer by actively listening and responding appropriately to their comments.

\n\n# Contextual Takeaways:\n\n

- **Repeating the same thing again without changing the language of the sentence**: Pitch the main aspects in at least eight different ways.
- **Ending call needs fixing**: Use friendly phrases like ""thank you"", ""sure"", ""that works"", ""see you"", etc.
- **AI is too pushy**: Ensure you are not pushy and respect the customer's responses.
- **Latency is too high**: Be responsive and minimize delays between responses.
- **Nodding is not there**: Use acknowledgments like ""hmm"", ""mmhmm"" when the prospect is speaking.
- **AI should engage with the customer**: Actively listen and interact appropriately throughout the conversation.

\n\n# Short Summary:\n\n

Implement these takeaways into your approach to ensure all points are covered and the conversation is natural, engaging, and effective.";
"""


@cl.on_chat_start
async def start():
    await cl.Message(content="Hi, is that Pranali with Twilio?").send()
    await setup_openai_realtime(system_prompt=system_prompt + "\n\n Customer ID: 12121")


@cl.on_message
async def on_message(message: cl.Message):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        await openai_realtime.send_user_message_content([{"type": 'input_text', "text": message.content}])
    else:
        await cl.Message(content="Please activate voice mode before sending messages!").send()


@cl.on_audio_start
async def on_audio_start():
    try:
        openai_realtime: RealtimeClient = cl.user_session.get(
            "openai_realtime")
        await openai_realtime.connect()
        logger.info("Connected to OpenAI realtime")
        return True
    except Exception as e:
        await cl.ErrorMessage(content=f"Failed to connect to OpenAI realtime: {e}").send()
        return False


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime:
        if openai_realtime.is_connected():
            await openai_realtime.append_input_audio(chunk.data)
        else:
            logger.info("RealtimeClient is not connected")


@cl.on_audio_end
@cl.on_chat_end
@cl.on_stop
async def on_end():
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        await openai_realtime.disconnect()
