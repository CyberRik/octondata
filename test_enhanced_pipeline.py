#!/usr/bin/env python3
"""
Test script for Enhanced Refined Pipeline
Tests improved image parsing and RAG integration
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_refined_pipeline import EnhancedRefinedPipeline
from rag_pipeline import RAGPipeline


def test_enhanced_image_parsing():
    """Test enhanced image parsing functionality"""
    print("🧪 Testing Enhanced Image Parsing...")
    
    try:
        # Initialize pipeline
        pipeline = EnhancedRefinedPipeline()
        
        # Test with sample PDF
        result = pipeline.process_document("sample.pdf")
        
        if result['status'] != 'success':
            print(f"❌ PDF processing failed: {result.get('error', 'Unknown error')}")
            return False
        
        # Check image extraction
        images = result.get('images', [])
        print(f"✓ PDF processing successful")
        print(f"  - Embedded images extracted: {len(images)}")
        
        # Validate image content
        for i, image in enumerate(images):
            print(f"  - Image {i+1}: {image['filename']}")
            print(f"    Content type: {image['content_type']}")
            print(f"    Confidence: {image['confidence']:.2f}")
            print(f"    Dimensions: {image['dimensions']}")
            print(f"    Is embedded: {image['is_embedded']}")
        
        # Check that we have reasonable number of images
        if len(images) > 0:
            print(f"✓ Found {len(images)} embedded images")
        else:
            print("⚠️  No embedded images found (this might be expected)")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced image parsing test failed: {e}")
        return False


def test_rag_integration():
    """Test RAG pipeline integration"""
    print("\n🧪 Testing RAG Integration...")
    
    try:
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        
        # Process document and build index
        result = rag_pipeline.process_document_and_build_index("sample.pdf")
        
        if result['status'] != 'success':
            print(f"❌ RAG index building failed: {result.get('error', 'Unknown error')}")
            return False
        
        print(f"✓ RAG index built successfully")
        print(f"  - Documents in index: {result['documents_count']}")
        print(f"  - Index path: {result['index_path']}")
        
        # Test querying
        test_queries = [
            "What is in the table?",
            "What images are in the document?",
            "What is the main content?"
        ]
        
        for query in test_queries:
            query_result = rag_pipeline.query(query)
            
            if query_result['status'] == 'success':
                print(f"✓ Query successful: '{query}'")
                print(f"  - Answer length: {len(query_result['answer'])}")
                print(f"  - Confidence: {query_result['confidence']:.3f}")
                print(f"  - Retrieved docs: {len(query_result['retrieved_documents'])}")
            else:
                print(f"❌ Query failed: '{query}' - {query_result.get('error', 'Unknown error')}")
                return False
        
        # Test index statistics
        stats = rag_pipeline.get_index_stats()
        if stats['status'] == 'success':
            print(f"✓ Index statistics retrieved")
            print(f"  - Total documents: {stats['total_documents']}")
            print(f"  - Type distribution: {stats['type_distribution']}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG integration test failed: {e}")
        return False


def test_embedded_image_detection():
    """Test specific embedded image detection"""
    print("\n🧪 Testing Embedded Image Detection...")
    
    try:
        pipeline = EnhancedRefinedPipeline()
        
        # Process document
        result = pipeline.process_document("sample.pdf")
        
        if result['status'] != 'success':
            print(f"❌ Document processing failed: {result.get('error', 'Unknown error')}")
            return False
        
        images = result.get('images', [])
        
        if not images:
            print("⚠️  No images found to test")
            return True
        
        # Test each image for embedded characteristics
        embedded_count = 0
        for image in images:
            is_embedded = image.get('is_embedded', False)
            confidence = image.get('confidence', 0.0)
            content_type = image.get('content_type', 'unknown')
            
            print(f"  - {image['filename']}:")
            print(f"    Embedded: {is_embedded}")
            print(f"    Confidence: {confidence:.2f}")
            print(f"    Content type: {content_type}")
            
            if is_embedded:
                embedded_count += 1
        
        print(f"✓ Found {embedded_count} embedded images out of {len(images)} total")
        
        # Check that we're not getting too many images (should be filtered)
        if len(images) <= 3:  # Reasonable number for a single PDF
            print("✓ Image count is reasonable (good filtering)")
        else:
            print(f"⚠️  High image count ({len(images)}), may need better filtering")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedded image detection test failed: {e}")
        return False


def test_output_organization():
    """Test output organization and file structure"""
    print("\n🧪 Testing Output Organization...")
    
    try:
        pipeline = EnhancedRefinedPipeline()
        
        # Process document
        result = pipeline.process_document("sample.pdf")
        
        if result['status'] != 'success':
            print(f"❌ Document processing failed: {result.get('error', 'Unknown error')}")
            return False
        
        # Check output directory structure
        output_dir = Path(result['output_directory'])
        
        required_dirs = ['images', 'tables', 'text']
        for dir_name in required_dirs:
            dir_path = output_dir / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.glob('*')))
                print(f"✓ {dir_name}: {file_count} files")
            else:
                print(f"❌ Missing directory: {dir_name}")
                return False
        
        # Check for processing summary
        summary_files = list(output_dir.glob('processing_summary_*.json'))
        if summary_files:
            print(f"✓ Processing summary: {len(summary_files)} file(s)")
        else:
            print("⚠️  No processing summary found")
        
        return True
        
    except Exception as e:
        print(f"❌ Output organization test failed: {e}")
        return False


def test_error_handling():
    """Test error handling for various edge cases"""
    print("\n🧪 Testing Error Handling...")
    
    try:
        pipeline = EnhancedRefinedPipeline()
        
        # Test with non-existent file
        result = pipeline.process_document("nonexistent.pdf")
        if result['status'] == 'error':
            print("✓ Error handling for nonexistent file working")
        else:
            print("❌ Should have failed for nonexistent file")
            return False
        
        # Test with empty images list
        empty_images = pipeline.extract_embedded_images({}, "test.pdf")
        if empty_images == []:
            print("✓ Empty images handling working")
        else:
            print("❌ Should return empty list for no images")
            return False
        
        # Test with empty tables
        empty_tables = pipeline.extract_tables({}, "test.pdf")
        if empty_tables == []:
            print("✓ Empty tables handling working")
        else:
            print("❌ Should return empty list for no tables")
            return False
        
        print("✓ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def cleanup_test_outputs():
    """Clean up test outputs"""
    print("\n🧹 Cleaning up test outputs...")
    
    try:
        # Remove test output directories
        test_dirs = ['./outputs', './rag_index.json']
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                if os.path.isdir(test_dir):
                    shutil.rmtree(test_dir)
                else:
                    os.remove(test_dir)
                print(f"  ✓ Removed {test_dir}")
        
        print("✓ Cleanup completed")
        return True
        
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Enhanced Pipeline Tests")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Enhanced Image Parsing", test_enhanced_image_parsing),
        ("RAG Integration", test_rag_integration),
        ("Embedded Image Detection", test_embedded_image_detection),
        ("Output Organization", test_output_organization),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    # Cleanup
    cleanup_test_outputs()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    for test_name, _ in tests:
        status = "✓ PASS" if test_name in [
            "Enhanced Image Parsing", "RAG Integration", "Embedded Image Detection", 
            "Output Organization", "Error Handling"
        ][:passed] else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced pipeline is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
