from openai import OpenAI
import tiktoken
import re
import os
import time


class LLMReranker:
    def __init__(self, llm_model="gpt-4o-mini"):
        self.llm_model = llm_model
        self.client = self._load_client()
        self.llm_attempts = 5

    # TODO: Add any additional models here
    def _load_client(self):
        if self.llm_model in ["gpt-4-turbo-preview", "gpt-4o-mini"]:
            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise ValueError("API key not found in environment variables.")

            return OpenAI(api_key=api_key)

    def num_tokens_from_string(self, string, encoding_name="gpt-4o-mini"):
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def rematch(
        self,
        source_table,
        target_table,
        source_values,
        target_values,
        matched_columns,
        score_based=True,
    ):
        refined_matches = {}
        column_count = 0
        for source_col, target_col_scores in matched_columns.items():
            # print(f"Refining matches for {source_col}", column_count, "out of", len(matched_columns))
            # column_count += 1
            cand = (
                "Column: "
                + source_col
                + ", Sample values: ["
                + ",".join(source_values[source_col])
                + "]"
            )
            target_cols = [
                "Column: "
                + target_col
                + ", Sample values: ["
                + ",".join(target_values[target_col])
                + "]"
                for target_col, _ in target_col_scores
            ]
            targets = "\n".join(target_cols)
            other_cols = ",".join(
                [col for col in source_table.columns if col != source_col]
            )
            if score_based:
                attempts = 0
                while True:

                    if attempts >= self.llm_attempts:
                        print(
                            f"Failed to parse response after {self.llm_attempts} attempts. Skipping."
                        )
                        refined_match = []
                        for target_col, score in target_col_scores:
                            refined_match.append((target_col, score))
                        break

                    refined_match = self._get_matches_w_score(cand, targets, other_cols)
                    refined_match = self._parse_scored_matches(refined_match)
                    attempts += 1
                    
                    if refined_match is not None:
                        break
                    


            else:
                refined_match = self._get_matches(cand, targets)
                refined_match = refined_match.split("; ")

            # print(f"Refined matches for {source_col}: {refined_match}")
            refined_matches[source_col] = refined_match
        return refined_matches

    def _get_prompt(self, cand, targets):
        prompt = (
            "From a score of 0.00 to 1.00, please judge the similarity of the candidate column from the candidate table to each target column in the target table. \
All the columns are defined by the column name and a sample of its respective values if available. \
Provide only the name of each target column followed by its similarity score in parentheses, formatted to two decimals, and separated by a semicolon. \
Rank the schema-score pairs by score in descending order. Ensure your response excludes additional information and quotations.\n \
Example:\n \
Candidate Column: \
Column: EmployeeID, Sample values: [100, 101, 102]\n \
Target Schemas: \
Column: WorkerID, Sample values: [100, 101, 102] \
Column: EmpCode, Sample values: [001, 002, 003] \
Column: StaffName, Sample values: ['Alice', 'Bob', 'Charlie']\n \
Response: WorkerID(0.95); EmpCode(0.30); StaffName(0.05)\n\n \
Candidate Column:"
            + cand
            + "\n\nTarget Schemas:\n"
            + targets
            + "\n\nResponse: "
        )
        # print('\n')
        # print(prompt)
        # print('\n')
        return prompt

    def _get_matches_w_score(
        self,
        cand,
        targets,
        other_cols,
    ):
        prompt = self._get_prompt(cand, targets)
        # print(prompt)
        if self.llm_model in ["gpt-4-turbo-preview", "gpt-4o-mini"]:
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI trained to perform schema matching by providing column similarity scores.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            # print(messages[1]["content"])

            # time_begin = time.time()

            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.3,
            )
            matches = response.choices[0].message.content

            # time_end = time.time()
            # print("Time taken for completion:", time_end - time_begin)

        elif self.llm_model in ["gemma2:9b"]:
            response = self.client.chat(
                model=self.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            matches = response["message"]["content"]
        # print(matches)
        return matches

    def _parse_scored_matches(self, refined_match):
        matched_columns = []
        entries = refined_match.split("; ")

        for entry in entries:
            try:
                schema_part, score_part = entry.rsplit("(", 1)
            except ValueError:
                print(f"Error parsing entry: {entry}")
                return None

            try:
                score = float(score_part[:-1])
            except ValueError:
                # Remove all trailing ')'
                score_part = score_part[:-1].rstrip(")")
                try:
                    score = float(score_part)
                except ValueError:
                    cleaned_part = re.sub(
                        r"[^\d\.-]", "", score_part
                    )  # Remove everything except digits, dot, and minus
                    match = re.match(r"^-?\d+\.\d{2}$", cleaned_part)
                    if match:
                        score = float(match.group())
                    else:
                        print("The string does not contain a valid two decimal float.")
                        return None

            schema_name = schema_part.strip()
            matched_columns.append((schema_name, score))

        return matched_columns
