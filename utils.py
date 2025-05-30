import os
import uuid
import requests
from urllib.parse import urlparse
from llama_index.llms.gemini import Gemini
import json
import re
import logging

logger = logging.getLogger(__name__)

def convert_string_to_json(json_string):
    # Remove markdown code blocks if present
    cleaned_string = re.sub(r'^```json\n?', '', json_string)
    cleaned_string = re.sub(r'\n?```$', '', cleaned_string)
    
    try:
        # Parse the JSON string
        json_object = json.loads(cleaned_string)
        return json_object
    except json.JSONDecodeError as error:
        print(f'Error parsing JSON: {error}')
        return None
    
    
    
def download_file_from_url(url: str, destination_path: str) -> str:
    """
    Download a file from URL and save it to destination path
    Returns the filename of the downloaded file
    """
    try:
        # Parse URL to get filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # If no filename in URL, generate one
        if not filename or '.' not in filename:
            filename = f"document_{uuid.uuid4().hex[:8]}.pdf"
        
        # Make HTTP request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        # Save file
        file_path = os.path.join(destination_path, filename)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
       
        return filename
        
    except requests.exceptions.RequestException as e:
        
        raise Exception(f"Failed to download file from {url}: {str(e)}")
    except Exception as e:
       
        raise Exception(f"Failed to download file from {url}: {str(e)}")


system_prompt = '''
           <SYSTEM_CONSTRAINTS>
                CRITICAL: Return EXACTLY ONE complete JSON object following the schema below.
                CRITICAL: Extract information from the ENTIRE PDF systematically.
                CRITICAL: Use "N/A" for any field where information is not found in the PDF.
                CRITICAL: Do NOT omit any fields from the schema - all fields must be present.
                CRITICAL: Return ONLY the JSON object - no additional text, explanations, or markdown.
          </SYSTEM_CONSTRAINTS>

<EXTRACTION_PROCESS>
1. Read through the entire PDF document completely
2. For each field in the schema, search the entire document for relevant information
3. If information is found, extract it exactly as written in the PDF
4. If information is not found, use "N/A" as the value
5. Always include source information when data is found
6. Return the complete JSON object with ALL fields populated
</EXTRACTION_PROCESS>

<JSON_SCHEMA>
You must return EXACTLY this JSON structure with ALL fields present:

{
  "dealOverview": {
    "source": {
      "page_no": "N/A or number",
      "start_position": "N/A or number",  
      "end_position": "N/A or number"
    },
    "propertyName": "N/A or extracted value",
    "location": "N/A or extracted value",
    "dateUploaded": "N/A or extracted value",
    "propertyType": "N/A or extracted value",
    "seller": "N/A or extracted value",
    "guidancePrice": "N/A or extracted value",
    "guidancePricePSF": "N/A or extracted value",
    "capRate": "N/A or extracted value",
    "propertySize": "N/A or extracted value",
    "landArea": "N/A or extracted value",
    "zoning": "N/A or extracted value",
    "underwritingModel": "N/A or extracted value"
  },
  "dealSummary": {
    "text": "N/A or extracted summary text"
  },
  "personalizedInsights": ["N/A or array of extracted insights"],
  "assetLevelData": {
    "source": {
      "page_no": "N/A or number",
      "start_position": "N/A or number",
      "end_position": "N/A or number"
    },
    "tenant": "N/A or extracted value",
    "clearHeights": "N/A or extracted value",
    "columnSpacing": "N/A or extracted value",
    "parkingSpaces": "N/A or number",
    "dockDoors": "N/A or number",
    "seawardArea": "N/A or extracted value",
    "yearBuilt": "N/A or number",
    "occupancyRate": "N/A or extracted value"
  },
  "projectedFinancialMetrics": {
    "IRR": "N/A or extracted value",
    "equityMultiple": "N/A or extracted value",
    "returnOnEquity": "N/A or extracted value",
    "returnOnCost": "N/A or extracted value"
  },
  "keyAssumptions": {
    "exitPrice": "N/A or extracted value",
    "exitCapRate": "N/A or extracted value",
    "rentalGrowth": "N/A or extracted value",
    "holdPeriod": "N/A or extracted value"
  },
  "marketAnalysis": {
    "source": {
      "page_no": "N/A or number",
      "start_position": "N/A or number",
      "end_position": "N/A or number"
    },
    "nearestUrbanCenter": "N/A or extracted value",
    "populationGrowthRate": "N/A or extracted value",
    "medianHouseholdIncome": "N/A or extracted value",
    "unemploymentRate": "N/A or extracted value"
  },
  "leaseAnalysis": {
    "rentPSF": "N/A or extracted value",
    "WALT": "N/A or extracted value",
    "rentEscalations": "N/A or extracted value",
    "markToMarketOpportunity": "N/A or extracted value"
  },
  "tenantDetails": {
    "tenant": {
      "source": {
        "page_no": "N/A or number",
        "start_position": "N/A or number",
        "end_position": "N/A or number"
      },
      "name": "N/A or extracted value",
      "logo": "/placeholder.svg?height=40&width=40",
      "industry": "N/A or extracted value",
      "creditRating": "N/A or extracted value"
    },
    "lease": {
      "source": {
        "page_no": "N/A or number",
        "start_position": "N/A or number",
        "end_position": "N/A or number"
      },
      "startDate": "N/A or YYYY-MM-DD",
      "expiryDate": "N/A or YYYY-MM-DD",
      "term": "N/A or extracted value",
      "remainingTerm": "N/A or extracted value"
    },
    "rent": {
      "source": {
        "page_no": "N/A or number",
        "start_position": "N/A or number",
        "end_position": "N/A or number"
      },
      "baseRentPSF": "N/A or extracted value",
      "annualBaseRent": "N/A or extracted value",
      "monthlyBaseRent": "N/A or extracted value",
      "effectiveRentPSF": "N/A or extracted value"
    },
    "escalations": {
      "structure": "N/A or extracted value",
      "rate": "N/A or extracted value",
      "nextEscalation": "N/A or extracted value"
    },
    "renewalOptions": [
      {
        "term": "N/A or extracted value",
        "notice": "N/A or extracted value",
        "rentStructure": "N/A or extracted value"
      }
    ],
    "recoveries": {
      "operatingExpenses": "N/A or extracted value",
      "cam": "N/A or extracted value",
      "insurance": "N/A or extracted value",
      "taxes": "N/A or extracted value",
      "utilities": "N/A or extracted value"
    },
    "security": {
      "deposit": "N/A or extracted value",
      "equivalent": "N/A or extracted value",
      "letterOfCredit": "N/A or extracted value"
    },
    "otherTerms": [
      {
        "title": "N/A or extracted value",
        "description": "N/A or extracted value"
      }
    ],
    "rentSchedule": [
      {
        "year": "N/A or number",
        "rentPSF": "N/A or number",
        "annualRent": "N/A or number"
      }
    ],
    "marketComparison": {
      "subjectProperty": { 
        "name": "N/A or extracted value", 
        "rentPSF": "N/A or number"
      },
      "marketComps": [
        { 
          "name": "N/A or extracted value", 
          "rentPSF": "N/A or number"
        }
      ]
    },
    "recoveryBreakdown": {
      "cam": "N/A or number",
      "taxes": "N/A or number",
      "insurance": "N/A or number"
    }
  }
}
</JSON_SCHEMA>

<CRITICAL_RULES>
1. Return ONLY the JSON object above - no additional text
2. ALL fields must be present in your response
3. Use "N/A" for missing information - do not omit fields
4. Maintain exact field names and structure as shown
5. Process the entire PDF before responding
6. Ensure valid JSON syntax (proper quotes, commas, brackets)
</CRITICAL_RULES>
        '''

def setup_llm_with_summary_prompt():
    return Gemini(
        model="models/gemini-2.0-flash",
    )

def setup_llm_with_source_prompt():
    try:
        llm = Gemini(
            model="models/gemini-2.5-flash-preview-05-20",
            system_prompt=
            '''
                <SYSTEM_CONSTRAINT>
                    ROLE: You are an expert in analyzing property documents and real estate data.

                    RESPONSE STYLE:
                         - Be brief and precise. Avoid any unnecessary elaboration.
                         - Only answer what is directly asked in the query. No extra context or assumptions.
                         - Use bullet points or tables wherever possible to structure the answer.
                         - Every fact must be immediately followed by a citation in this format: [Source: filename.pdf, Page X].

                    CITATION RULE:
                         - No fact should appear without a citation.
                         - Example: "Zoning: R-1 Residential [Source: zoning_report.pdf, Page 3]"

                    CONTENT SCOPE:
                          - Do NOT provide any information not found in the property documents.
                          - Only include content relevant to the specific query.
                          
                    FORMATTING:
                         - Use markdown for bullets and tables.
                </SYSTEM_CONSTRAINT>
            ''',
            temperature=0.1,  # Lower temperature for more focused responses
            max_tokens=1000,  # Ensure we get substantial responses
            top_p=0.9,  # Slightly more focused sampling
            top_k=40  # Good balance between diversity and focus
        )
        return llm
    except Exception as e:
        logger.error(f"Error setting up LLM: {str(e)}")
        raise