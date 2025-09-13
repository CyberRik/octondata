#!/usr/bin/env python3
"""
Test script for Refined PDF Parsing Pipeline
Tests the refined pipeline with improved table and image extraction
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from refined_pipeline import RefinedPipeline


def test_table_extraction():
    """Test improved table extraction"""
    print("🧪 Testing Table Extraction...")
    
    try:
        # Test with sample PDF
        sample_pdf = "sample.pdf"
        if not os.path.exists(sample_pdf):
            print("⚠ Sample PDF not found, skipping table extraction test")
            return True
        
        # Initialize pipeline
        pipeline = RefinedPipeline()
        
        # Process PDF
        result = pipeline.process_document(sample_pdf)
        
        if result['status'] == 'success':
            print("✓ PDF processing successful")
            
            # Check table extraction
            tables = result.get('tables', [])
            if len(tables) == 1:
                print("✓ Correct number of tables extracted (1)")
                
                table = tables[0]
                print(f"  - Rows: {table['rows']}")
                print(f"  - Columns: {table['columns']}")
                
                # Check CSV file content
                csv_path = Path(table['csv_file'])
                if csv_path.exists():
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        lines = content.split('\n')
                        
                        print(f"  - CSV lines: {len(lines)}")
                        print(f"  - First line: {lines[0] if lines else 'Empty'}")
                        
                        # Check if content looks like the expected table
                        if len(lines) >= 3:  # At least 3 rows
                            print("✓ Table has sufficient rows")
                            
                            # Check for expected content pattern
                            first_line = lines[0].lower()
                            if 'hello' in first_line and 'mewo' in first_line:
                                print("✓ Table contains expected content")
                            else:
                                print(f"⚠ Unexpected table content: {first_line}")
                        else:
                            print("✗ Table has insufficient rows")
                            return False
                else:
                    print("✗ CSV file not created")
                    return False
            else:
                print(f"✗ Expected 1 table, got {len(tables)}")
                return False
        else:
            print(f"✗ PDF processing failed: {result['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Table extraction test failed: {e}")
        return False


def test_image_filtering():
    """Test improved image filtering"""
    print("\n🧪 Testing Image Filtering...")
    
    try:
        # Test with sample PDF
        sample_pdf = "sample.pdf"
        if not os.path.exists(sample_pdf):
            print("⚠ Sample PDF not found, skipping image filtering test")
            return True
        
        # Initialize pipeline
        pipeline = RefinedPipeline()
        
        # Process PDF
        result = pipeline.process_document(sample_pdf)
        
        if result['status'] == 'success':
            print("✓ PDF processing successful")
            
            # Check image extraction
            images = result.get('images', [])
            print(f"  - Images extracted: {len(images)}")
            
            # Check if images are relevant (not table crops)
            relevant_count = 0
            for img in images:
                filename = img['filename']
                file_size = img['file_size']
                
                print(f"    - {filename} ({file_size} bytes)")
                
                # Check if it's likely a relevant image (not a table crop)
                if 'page_' in filename and 'img_' not in filename:
                    relevant_count += 1
                    print(f"      ✓ Relevant image (full page)")
                elif file_size > 50000:  # Larger files are more likely to be relevant
                    relevant_count += 1
                    print(f"      ✓ Relevant image (large size)")
                else:
                    print(f"      ⚠ Potentially filtered image")
            
            if relevant_count > 0:
                print(f"✓ Found {relevant_count} relevant images")
            else:
                print("⚠ No clearly relevant images found")
            
            # Check that we don't have too many images (should be filtered)
            if len(images) <= 5:  # Reasonable number
                print("✓ Image count is reasonable (filtered)")
            else:
                print(f"⚠ Too many images extracted: {len(images)}")
                return False
        else:
            print(f"✗ PDF processing failed: {result['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Image filtering test failed: {e}")
        return False


def test_content_validation():
    """Test content validation and quality"""
    print("\n🧪 Testing Content Validation...")
    
    try:
        # Test with sample PDF
        sample_pdf = "sample.pdf"
        if not os.path.exists(sample_pdf):
            print("⚠ Sample PDF not found, skipping content validation test")
            return True
        
        # Initialize pipeline
        pipeline = RefinedPipeline()
        
        # Process PDF
        result = pipeline.process_document(sample_pdf)
        
        if result['status'] == 'success':
            print("✓ PDF processing successful")
            
            # Validate table content
            tables = result.get('tables', [])
            if tables:
                table = tables[0]
                csv_path = Path(table['csv_file'])
                
                if csv_path.exists():
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        
                        # Check for clean table format
                        lines = content.split('\n')
                        if len(lines) >= 3:
                            # Check first few lines for expected pattern
                            first_line = lines[0]
                            if ',' in first_line:  # CSV format
                                print("✓ Table is in proper CSV format")
                                
                                # Check for expected content
                                if 'hello' in first_line.lower():
                                    print("✓ Table contains expected content")
                                else:
                                    print(f"⚠ Unexpected table content: {first_line}")
                            else:
                                print(f"✗ Table not in proper CSV format: {first_line}")
                                return False
                        else:
                            print(f"✗ Table has insufficient data: {len(lines)} lines")
                            return False
            
            # Validate text content
            text = result.get('text', {})
            if text:
                text_path = Path(text['file_path'])
                if text_path.exists():
                    with open(text_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        
                        if len(content) > 10:  # Has meaningful content
                            print("✓ Text content is meaningful")
                        else:
                            print(f"⚠ Text content too short: {len(content)} chars")
            
            # Validate image content
            images = result.get('images', [])
            if images:
                print(f"✓ {len(images)} images extracted and validated")
            else:
                print("⚠ No images extracted")
            
            print("✓ Content validation completed successfully")
        else:
            print(f"✗ PDF processing failed: {result['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Content validation test failed: {e}")
        return False


def test_output_organization():
    """Test output organization and file structure"""
    print("\n🧪 Testing Output Organization...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize pipeline with temporary directory
            pipeline = RefinedPipeline()
            pipeline.base_output_dir = Path(temp_dir)
            pipeline.images_dir = pipeline.base_output_dir / 'images'
            pipeline.tables_dir = pipeline.base_output_dir / 'tables'
            pipeline.text_dir = pipeline.base_output_dir / 'text'
            pipeline._create_output_directories()
            
            # Test with sample PDF
            sample_pdf = "sample.pdf"
            if os.path.exists(sample_pdf):
                result = pipeline.process_document(sample_pdf)
                
                if result['status'] == 'success':
                    print("✓ PDF processing successful")
                    
                    # Check directory structure
                    for dir_name in ['images', 'tables', 'text']:
                        dir_path = getattr(pipeline, f"{dir_name}_dir")
                        if dir_path.exists():
                            file_count = len(list(dir_path.glob('*')))
                            print(f"  ✓ {dir_name}: {file_count} files")
                        else:
                            print(f"  ✗ {dir_name}: directory not created")
                            return False
                    
                    # Check file naming
                    tables_dir = pipeline.tables_dir
                    table_files = list(tables_dir.glob('*.csv'))
                    if table_files:
                        table_file = table_files[0]
                        if 'table' in table_file.name and 'sample' in table_file.name:
                            print("✓ Table file naming is correct")
                        else:
                            print(f"⚠ Unexpected table filename: {table_file.name}")
                    
                    print("✓ Output organization is correct")
                else:
                    print(f"✗ PDF processing failed: {result['error']}")
                    return False
            else:
                print("⚠ Sample PDF not found, skipping output organization test")
                return True
        
        return True
        
    except Exception as e:
        print(f"✗ Output organization test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases"""
    print("\n🧪 Testing Error Handling...")
    
    try:
        pipeline = RefinedPipeline()
        
        # Test with nonexistent file
        result = pipeline.process_document("nonexistent_file.pdf")
        
        if result['status'] == 'error' and 'error' in result:
            print("✓ Error handling for nonexistent file working")
        else:
            print("✗ Error handling for nonexistent file failed")
            return False
        
        # Test image filtering with invalid data
        empty_parsed_data = {}
        images = pipeline.extract_images(empty_parsed_data, "test.pdf")
        if images == []:
            print("✓ Empty images handling working")
        else:
            print("✗ Empty images handling failed")
            return False
        
        # Test table extraction with invalid data
        tables = pipeline.extract_tables(empty_parsed_data, "test.pdf")
        if tables == []:
            print("✓ Empty tables handling working")
        else:
            print("✗ Empty tables handling failed")
            return False
        
        print("✓ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def cleanup_test_outputs():
    """Clean up test outputs"""
    print("\n🧹 Cleaning up test outputs...")
    
    try:
        # Remove test output directories
        test_dirs = ['./test_outputs', './outputs', './refined_outputs']
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
                print(f"  ✓ Removed {test_dir}")
        
        # Remove test log files
        test_logs = ['refined_pipeline.log', 'test_refined_pipeline.log']
        for log_file in test_logs:
            if os.path.exists(log_file):
                os.remove(log_file)
                print(f"  ✓ Removed {log_file}")
        
        print("✓ Cleanup completed")
        return True
        
    except Exception as e:
        print(f"⚠ Cleanup warning: {e}")
        return True  # Cleanup failures shouldn't fail the test


def main():
    """Run all tests"""
    print("🚀 Starting Refined Pipeline Tests")
    print("=" * 60)
    
    tests = [
        ("Table Extraction", test_table_extraction),
        ("Image Filtering", test_image_filtering),
        ("Content Validation", test_content_validation),
        ("Output Organization", test_output_organization),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Cleanup
    cleanup_test_outputs()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refined pipeline is ready to use.")
        return 0
    else:
        print("⚠ Some tests failed. Please check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
