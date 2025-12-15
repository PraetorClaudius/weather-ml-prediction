import boto3
import json

s3_client = boto3.client('s3')

raw_bucket = 'weather-raw-data-lalo'
processed_bucket = 'weather-processed-data-lalo'

# Get all raw files
response = s3_client.list_objects_v2(Bucket=raw_bucket, Prefix='raw/')
files = response.get('Contents', [])

print(f"Found {len(files)} raw files to process")

for idx, file in enumerate(files, 1):
    key = file['Key']
    print(f"Processing {idx}/{len(files)}: {key}")
    
    try:
        # Read raw file
        obj = s3_client.get_object(Bucket=raw_bucket, Key=key)
        raw_data = json.loads(obj['Body'].read().decode('utf-8'))
        
        # Process data (same logic as your Lambda)
        processed_records = []
        for record in raw_data:
            processed_record = {
                'city': record.get('name'),
                'country': record.get('sys', {}).get('country'),
                'timestamp': record.get('collection_timestamp'),
                'temperature_celsius': record.get('main', {}).get('temp'),
                'feels_like': record.get('main', {}).get('feels_like'),
                'humidity_percent': record.get('main', {}).get('humidity'),
                'pressure_hpa': record.get('main', {}).get('pressure'),
                'wind_speed_mps': record.get('wind', {}).get('speed'),
                'weather_description': record.get('weather', [{}])[0].get('description'),
                'cloudiness_percent': record.get('clouds', {}).get('all'),
                'latitude': record.get('coord', {}).get('lat'),
                'longitude': record.get('coord', {}).get('lon')
            }
            processed_records.append(processed_record)
        
        # Create processed key (mirror the structure)
        processed_key = key.replace('raw/', 'processed/').replace('weather_', 'weather_processed_')
        
        # Save to processed bucket
        s3_client.put_object(
            Bucket=processed_bucket,
            Key=processed_key,
            Body=json.dumps({'records': processed_records}, indent=2),
            ContentType='application/json'
        )
        
        print(f"  Saved to {processed_key}")
        
    except Exception as e:
        print(f"  Error: {e}")

print(f"\nDone! Processed {len(files)} files")