# Data Source

The entire dataset disclosure application is in progress and will be released later.


# Data Format

### user_behavior_history.csv
The data set covers all behaviors (including ID, call time, status, and log) of approximately 30 million random users from January 1, 2023 to June 30, 2024. The organization of the TelM-1 dataset and the TelM-2 dataset is similar. Each sample in the original log data contains the User ID, call time, connection status, call duration, etc. The detailed description of each column in the data set is as follows:

| Field | Explanation |
| --- | --- |
| User ID | An integer, the serialized ID that represents a user |
| Call time | An integer, the timestamp of the behavior |
| Call status | A string, enum-type from ('answer', 'unanswer') |
| Call duration | A float, the call duration of the behavior |


### user_profile.csv
These two data sets mainly contains the basic attributes of about 30 million random users, such as age, gender, occupation, resident city ID, etc. Each row of the data set represents a piece of user information, separated by commas. The detailed description of each column in the data set is as follows:
| Field | Explanation |
| --- | --- |
| User ID | An integer, the serialized ID that represents a user |
| Age | An integer, the serialized ID that represents an user age |
| Gender | An integer, the serialized ID that represents the user gender |
| Occupation | An integer, the serialized ID that represents the user occupation |
| Habitual City | An integer, the serialized ID that represents the user habitual city，single value |

### historical_connection_profile.csv
This data file mainly contains the basic attribute characteristics of about 317 features, such as Time slot, Preference label, Call times, Connection times, Connection ratio etc. Each row of the data set represents the information of a user, separated by commas. The detailed description of each column in the data set is as follows:
| Field | Explanation |
| --- | --- |
| User ID | An integer, the serialized ID that represents an item |
| Time slot | An integer, enum-type from (9, 10, 11, 12, 14, 15, 16, 17) |
| Preference label | An integer, whether user has a preference for time slot, enum-type from (0,1)  |
| Call times 3d | An integer, The number of times the user was called in the past 3 days |
| Call times 1m | An integer, The number of times the user was called in the past 30 days |
| Connection times 3d | An integer, The number of times the user was answered in the past 3 days |
| Connection times 1m | An integer, The number of times the user was answered in the past 30 days |
| Time slot connection ratio 1m| An float, The call connection rate for the user within this time slot over the past 30 days |
| h9 connection ratio 1m| An float, The call connection rate for the user within time slot is equal to 9 over the past 30 days |
| h10 connection ratio 1m| An float, The call connection rate for the user within time slot is equal to 10 over the past 30 days |
| h11 connection ratio 1m| An float, The call connection rate for the user within time slot is equal to 11 over the past 30 days |
| h12 connection ratio 1m| An float, The call connection rate for the user within time slot is equal to 12 over the past 30 days |
| h14 connection ratio 1m| An float, The call connection rate for the user within time slot is equal to 14 over the past 30 days |
| h15 connection ratio 1m| An float, The call connection rate for the user within time slot is equal to 15 over the past 30 days |
| h16 connection ratio 1m| An float, The call connection rate for the user within time slot is equal to 16 over the past 30 days |
| h17 connection ratio 1m| An float, The call connection rate for the user within time slot is equal to 17 over the past 30 days |

