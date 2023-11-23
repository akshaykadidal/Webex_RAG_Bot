import os
from gpt import gpt

from webex_bot.webex_bot import WebexBot


# Create a Bot Object
bot = WebexBot(teams_bot_token=os.getenv('WEBEX'))

#clear defaults
bot.commands.clear()

# Add new commands for the bot to listen out for.
bot.add_command(gpt())

#set new default
bot.help_command = gpt()

# Call `run` for the bot to wait for incoming messages.
bot.run()
