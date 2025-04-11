from crewai import Task

class DiseasePredictorTask:
    def create_disease_researcher_task(self, agent, disease,specific_aspects=None):
        
        aspects = specific_aspects if specific_aspects else [
            "etiology",
            "pathophysiology",
            "clinical manifestations",
            "diagnostic criteria",
            "treatment options",
            "prevention strategies"
        ]
        
        aspects_str = ", ".join(aspects)

        # Research Task
        return Task(
            name="Disease Research",
                description=f'''
                Conduct a thorough investigation of {disease} with focus on: {aspects_str}
                
                Required analysis components:
                1. Disease Overview:
                   - Definition and classification
                   - Historical context and significance
                   - Epidemiology and prevalence data
                
                2. Clinical Analysis:
                   - Detailed pathophysiology
                   - Disease progression stages
                   - Common and rare manifestations
                   - Complications and comorbidities
                
                3. Diagnostic Approach:
                   - Primary diagnostic criteria
                   - Differential diagnoses
                   - Laboratory and imaging findings
                   - Diagnostic challenges
                
                4. Treatment Evaluation:
                   - Current treatment protocols
                   - Evidence-based interventions
                   - Treatment efficacy data
                   - Management of complications
                
                5. Prevention and Prognosis:
                   - Risk factors identification
                   - Preventive strategies
                   - Long-term outcomes
                   - Quality of life impact
            
            Inputs:
            Disease Name: {disease}
            
            ''',
            expected_output=f'''
                Deliver a comprehensive report on {disease} including:
                
                1. Executive Summary
                2. Detailed analysis of each required component
                3. Evidence-based findings and recommendations
                4. Clinical implications and practical applications
                5. Future research directions and gaps in current knowledge
                6. References to key studies and guidelines
                
                Format: Structured report with clear sections, supporting data, and actionable insights.
                Length: Comprehensive coverage of all aspects while maintaining clarity and relevance.
                Evidence: Include references to current research and clinical guidelines.
            ''',
            agent=agent,
            verbose=True
        )
