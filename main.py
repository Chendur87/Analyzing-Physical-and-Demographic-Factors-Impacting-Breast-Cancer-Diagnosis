"""
Chendur Jayavelu
This module is runs all of the Research Questions
"""

from research_question_one import ResearchQuestionOne
from testing import research_question_one_testing as rq_one_testing
from research_question_two import ResearchQuestionTwo
from testing import research_question_two_testing as rq_two_testing
from research_question_three import ResearchQuestionThree
import time


def main():
    print('Starting Research Question #1')
    start_rq_one = time.time()
    rq_one = ResearchQuestionOne()
    rq_one.run()
    rq_one_testing.testing()
    end_rq_one = time.time()
    print('\nResearch Question #1 complete')
    time_one = end_rq_one - start_rq_one
    print(f"Time for RQ #1: {time_one} seconds")

    start_rq_two = time.time()
    rq_two = ResearchQuestionTwo()
    rq_two.run()
    rq_two_testing.testing()
    end_rq_two = time.time()
    print('\nResearch Question #2 complete')
    time_two = end_rq_two - start_rq_two
    print(f"Time for RQ #2: {time_two} seconds")

    start_rq_three = time.time()
    rq_three = ResearchQuestionThree()
    rq_three.run()
    end_rq_three = time.time()
    print('\nResearch Question #3 complete')
    time_three = end_rq_three - start_rq_three
    print(f"Time for RQ #3: {time_three} seconds")

    total_time = time_one + time_two + time_three
    print(f"Total time taken: {total_time} seconds")


if __name__ == '__main__':
    main()