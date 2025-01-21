import pandas as pd
import numpy as np
import re
import random

# set API keys

# set prompts
instruction_prompt = """
                      # Instruction
                      You will be given challenging work situations. For each situation, you will be provided possible responses (a to d). Your job is to consider and
                      think through the work situation step-by-step and then choose the best and the worst responses from the possible options (a to d).
                      Self-evaluate your confidence in your response after you answer. Provide an output confidence estimate ranging from 0 to 100%.
                      Use the following output format:
                      ## Output Format
                      [full response goes here]
                      Confidence: XX%
                      Best Option: X
                      Worst Option: Y
                      """

sjt_items = [
              # item 1
              """ You are temporarily subjected to personal stress that also affects your occupational activity. A briefly acquainted colleague asks you about the reason for your decline in performance and offers help with your task. What should you do and not do in such a situation?\n\n
              a)	You confirm an increase in personal stress and accept the help.
              b)	You do not mention your personal problems, but accept the help.
              c)	You explain your specific situation and ask for help with your task.
              d)	You thank your colleague for the feedback but politely decline the offer.""",
              # item 2
              """Your team has clearly allocated all areas of responsibility. However, you incidentally notice that some team members in another area are challenged by a task that you have experience in.What should you do and not do in such a situation?\n\n
              a)	You tell your colleagues about your experience and offer advisory support.
              b)	You mention your expertise and offer active support.
              c)	You ask whether your experience or advice is desired and if so, when.
              d)	Your respect your colleagues’ responsibilities and stay out of it.""",
              # item 3
              """Your team has made a lot of progress working on a complex task when some unforeseen developments occur. Therefore, your tediously achieved results are no longer completely up to date.What should your team do and not do in such a situation?
              a)	Slight shortcomings will be tolerated due to the advanced progress of the work.
              b)	The team asks the customer or superior for their assessment of the situation.
              c)	Team members immediately discuss possible consequences in a meeting.
              d)	Changes are retrospectively implemented through intensive additional work.""",
              # item 4
              """ You notice a sudden but continuous decrease in performance in one of your team members, whom you have experienced as competent and reliable. Other sources report that this colleague currently has some personal problems.
              What should you do and not do in such a situation?
              a)	You and the other team members discuss how to support this colleague.
              b)	You respect your colleague’s privacy and do not get involved in private matters.
              c)	You help your colleague without asking questions.
              d)	You ask your colleague if they want to talk about their problems.""",
              # item 5
              """Some of your colleagues discuss various aspects of a team task during a meeting. Your area of responsibility is not the focus of their discussion, which is why you hold back and refrain from partaking in it.
              What should you do and not do in such a situation?
              a)	You mentally prepare the discussion points that you wanted to address.
              b)	You use the opportunity to broaden your knowledge about other parts of the task.
              c)	You carefully steer the conversation towards a more familiar topic that you can engage in.
              d)	You attentively look for information that could be important for your area of responsibility.""",
              # item 6
              """You are transferred to an already existing team. During a brief introduction, your contact person tells you that all team members have their own area of responsibility. Without providing any further details, your contact person instructs you on your own area of responsibility.
              What should you do and not do in such a situation?
              a)	You limit your questions to your area of responsibility, as nothing else should concern you.
              b)	You hold back your curiosity and carefully listen to your contact person’s explanations.
              c)	You decide to become familiar with the other areas of responsibility on your own after the conversation.
              d)	You ask for the basic workflows and interdependencies in the team.""",
              # item 7
              """You have to inform another team member about a complex issue from your area of responsibility. It is of utmost importance for your team’s success that the other person takes note of your concern and that no uncertainties are left.
              What should you do and not do in such a situation?
              a)	You prepare a timesaving summary that you personally deliver.
              b)	You send a detailed report and ask for an acknowledgement of receipt.
              c)	You arrange a personal meeting with the other team member.
              d)	You send a message and explicitly request the other team member to contact you if any uncertainties are left.""",
              # item 8
              """You incidentally notice that another team member struggles to finish their work on time. You have already completed your tasks. However, you want to double check your work and, if necessary, improve some details before the deadline.
              What should you do and not do in such a situation?
              a)	You ask the team member in a confidential conversation whether they need help.
              b)	You contain yourself because you do not want the team member to appear incompetent.
              c)	You carefully address your observation in the next team meeting.
              d)	You finish your own tasks first before offering your help.""",
              # item 9
              """Together with your team members, you are setting objectives for each member for an upcoming task.
              What should your team do and not do in such a situation?
              a)	The team sets objectives that are positive, clearly defined and easily verifiable.
              b)	The team sets objectives that are specific, challenging and agreed upon by the whole team.
              c)	The team sets objectives that are moderately difficult and comprehensible to the whole team.
              d)	The team sets objectives that are easily attainable, open and flexible concerning time management.""",
              # item 10
              """You are working on a task that is mainly in your area of responsibility. When you present your intended procedure during a meeting, some team members from other areas speak up and add suggestions for changes and adaptations.
              What should you do and not do in such a situation?
              a)	You take note of suggestions and discuss them with everyone involved.
              b)	You reflect on what changes might be sensible and ask for details.
              c)	You politely point out that you have a better overview of the task due to your expertise.
              d)	You try to include as many of the suggested changes as possible in your plan.""",
              # item 11
              """Due to external circumstances, your team was unable to finish an important task on time. Since every team member has given their very best, there is considerable disappointment. When a new task comes up, you notice that low morale and poor motivation are impairing the team.
              What should you do and not do in such a situation?
              a)	You remind your team of past successes to spark new motivation.
              b)	You address your concerns in front of the whole team and encourage a discussion.
              c)	You ask for a team meeting to put the failure behind you.
              d)	You give the other team members the time to regain their motivation.""",
              # item 12
              """Together with your team members, you are planning how to tackle an upcoming task. The team’s success in mastering this challenge depends on several factors, some of which are difficult to predict.
              What should your team do and not do in such a situation?
              a)	The team discusses all possible developments in advance and works out a strategy for each of them.
              b)	The planning proceeds in small steps in order to allow quick adaptations.
              c)	The team waits with further planning until all uncertainties are eliminated.
              d)	The team focuses especially on currently available facts for the planning."""
              ]

### RUN PILOT ###
### HUGGINGFACE API
### LLAMA 8B

# initialize
results_df = pd.DataFrame(columns=['response', 'logprobs', 'certainty', 'perplexity', 'avg_linear_prob', 'entropy'])

# set random seed
random.seed(514)

# loop API call
for sjt_item in sjt_items:
    seed_number = random.randint(1, 10000)
    completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct", # need hf pro for 70B
        messages=[{"role": "system", "content": instruction_prompt},
                  {"role": "user", "content":f"""Work situation:{sjt_item}"""}],
        seed=seed_number,
        logprobs=True,
        top_logprobs = 3,
    )

    # extract response text
    response_text = completion.choices[0].message.content

    # identify certainty score
    certainty_match = re.search(r'Confidence:\s*(\d+)%', response_text)
    certainty = float(certainty_match.group(1)) if certainty_match else None

    # identify best and worst match
    best_match = re.search(r'Best Option:\s*([a-d])', response_text)
    worst_match = re.search(r'Worst Option:\s*([a-d])', response_text)

    # extract tokens
    token_logprobs = [lp.logprob for lp in completion.choices[0].logprobs.content]
    token_top_logprobs = [lp.top_logprobs for lp in completion.choices[0].logprobs.content]

    # calculate perplexity (np.exp(-np.mean(tokens_df['logprob'])))
    avg_logprob = np.mean(token_logprobs)
    perplexity = np.exp(-avg_logprob)

    # calc avg linear probability (tokens_df['probability'] = np.exp(tokens_df['logprob']))
    linear_probs = np.exp(token_logprobs)  # convert from log space to probability space
    avg_linear_prob = np.mean(linear_probs)

    # calc entropy (entropy = -np.sum([p * np.log2(p) for p in linear_probs if p > 0])
    # entropy = -np.sum([p * np.log2(p) for p in linear_probs if p > 0])
    entropy = 0
    for i in range(len(token_logprobs)):
        # Get all probabilities for this position (main token + alternatives)
        main_prob = np.exp(token_logprobs[i])
        alt_probs = [np.exp(alt.logprob) for alt in token_top_logprobs[i]]
        all_probs = [main_prob] + alt_probs

        # Calculate position entropy
        position_entropy = -np.sum([p * np.log2(p) for p in all_probs if p > 0])
        entropy += position_entropy

    # Average entropy over all positions
    entropy = entropy / len(token_logprobs)

    # create new rows
    new_row = pd.DataFrame({
        'response': [completion.choices[0].message.content],
        'logprobs': [completion.choices[0].logprobs],
        'certainty': [certainty],
        'perplexity': [perplexity],
        'avg_linear_prob': [avg_linear_prob],
        'entropy': [entropy]
    })

    # Append to results
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    print(completion.choices[0].message.content)
    print("\n### NEXT ITEM ###\n")
