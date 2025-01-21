

initial_setup = """
You are participating in a research study exploring interests and preferences in a virtual environment. You will be placed in a circular room containing six different work areas. You can freely explore these areas, engage with available activities, and choose how to spend your time. You can move between areas, engage more deeply with activities that interest you, or exit the study at any time. Please respond naturally, describing your thoughts, choices, and activities.

You enter the room and see the following:

## Room Setup ##

You find yourself in a circular room with soft, ambient lighting. Six distinct work areas are arranged around the perimeter, each equally accessible from your current position in the center. Each area has its own character and available activities. You can clearly see all areas from where you stand and you can move and interact freely between and with them.

[Technical Area]
A streamlined workspace with multiple high-resolution displays showing code, system diagrams, and technical documentation. The desk contains diagnostic tools and engineering problems ready for analysis. The space has a precise, organized feel, with tools arranged systematically and clear procedures for technical problem-solving.

[Research Space]
A thought-provoking space with interactive data visualizations, research materials, and scientific equipment. Multiple screens display evolving data patterns and experimental results. The area invites analytical thinking and systematic investigation, with resources for deep theoretical exploration.

[Creative Studio]
An inspiring workspace filled with diverse creative prompts and artistic challenges. The area features various forms of artistic expression and design opportunities. The space encourages original thinking and aesthetic exploration, with materials for creative development and artistic analysis.

[Communication Hub]
A welcoming space focused on human interaction and understanding. The area features scenarios for teaching, counseling, and community problem-solving. Resources for cultural analysis and relationship dynamics are readily available, encouraging helpful and insightful communication.

[Strategy Center]
A dynamic workspace for decision-making and leadership challenges. Displays show market trends, organizational structures, and resource allocation scenarios. The space is designed for strategic thinking and project development, with tools for analysis and planning.

[Systems Hub]
A precisely organized area focused on information management and process optimization. Screens display data structures, workflow diagrams, and system architectures. The space emphasizes systematic thinking and efficient organization, with tools for developing and improving processes.

[The Door]
Behind you is the door that you just entered to get in the room. Interacting with the door again will allow you to leave the room and exit the study.

## Movement and Interaction Controls ##

You can:
- Move to any area that interests you
- Observe and explore available activities
- Engage with tasks at different levels of complexity
- Return to previously visited areas
- Move to a different area at any time
- Exit the study when you wish

As you explore, you can engage with activities at increasing levels of depth:
- [Level 1] Initial exploration and observation
- [Level 2] Active participation and task engagement
- [Level 3] Complex problem-solving and creative development

## Start ##
What would you like to do first?
"""

# API calls

class FreePlayConversation:  
    def _ _init_ _(self, initial_setup):  
        self.messages = [  
            {"role": "system", "content": initial_setup}  
        ]  
          
    def add_exchange(self, client):  
    """ make API sequential API calls with accumulating messages"""  
        try:  
            # huggingface serverless inference format  
            completion = client.chat.completions.create(  
                model="LLM", # set model name here  
                messages=self.messages + [  
                    {"role": "user", "content": "What do you do next?"}  
                ]  
            )  
              
            # extract response  
            response = completion.choices[0].message.content  
              
            # append for message history  
            self.messages.append(  
                {"role": "user", "content": "What do you do next?"}  
            )  
            self.messages.append(  
                {"role": "assistant", "content": response}  
            )  
              
            return response  
              
        except Exception as e:  
            print(f"Error in API call: {e}")  
            return None  
