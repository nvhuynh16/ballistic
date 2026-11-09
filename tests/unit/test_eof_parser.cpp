// Unit tests for the EOF parser. An EOF (Earth Explorer Orbit File) is ESA's XML
// format for orbit products such as the Sentinel precise orbits (POE); it holds a
// validity period and a list of orbit state vectors (OSVs: time, position,
// velocity). The checks are exact parses of hand-written values, refusal or
// skipping of malformed input, and the time queries.
// Uses synthetic inline XML — no external data dependency

#include "../test_utils.hpp"
#include <ballistic/validation/eof_parser.hpp>
#include <cmath>
#include <filesystem>
#include <string>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::validation;

// =============================================================================
// Synthetic XML helpers
// =============================================================================

static const std::string VALID_2OSV_XML = R"(<?xml version="1.0"?>
<Earth_Explorer_File>
  <Earth_Explorer_Header>
    <Fixed_Header>
      <Mission>Sentinel-1A</Mission>
      <Validity_Period>
        <Validity_Start>UTC=2024-01-15T00:00:00.000000</Validity_Start>
        <Validity_Stop>UTC=2024-01-15T23:59:59.000000</Validity_Stop>
      </Validity_Period>
    </Fixed_Header>
  </Earth_Explorer_Header>
  <Data_Block>
    <List_of_OSVs count="2">
      <OSV>
        <TAI>TAI=2024-01-15T12:00:37.000000</TAI>
        <UTC>UTC=2024-01-15T12:00:00.000000</UTC>
        <UT1>UT1=2024-01-15T11:59:59.900000</UT1>
        <Absolute_Orbit>52001</Absolute_Orbit>
        <X unit="m">4500000.000</X>
        <Y unit="m">3500000.000</Y>
        <Z unit="m">4000000.000</Z>
        <VX unit="m/s">-2500.000</VX>
        <VY unit="m/s">5000.000</VY>
        <VZ unit="m/s">5500.000</VZ>
        <Quality>NOMINAL</Quality>
      </OSV>
      <OSV>
        <TAI>TAI=2024-01-15T12:00:47.000000</TAI>
        <UTC>UTC=2024-01-15T12:00:10.000000</UTC>
        <UT1>UT1=2024-01-15T12:00:09.900000</UT1>
        <Absolute_Orbit>52001</Absolute_Orbit>
        <X unit="m">4499975.000</X>
        <Y unit="m">3500050.000</Y>
        <Z unit="m">4000055.000</Z>
        <VX unit="m/s">-2500.100</VX>
        <VY unit="m/s">5000.200</VY>
        <VZ unit="m/s">5499.900</VZ>
        <Quality>NOMINAL</Quality>
      </OSV>
    </List_of_OSVs>
  </Data_Block>
</Earth_Explorer_File>
)";

static const std::string VALID_WITH_REF_FRAME_XML = R"(<?xml version="1.0"?>
<Earth_Explorer_File>
  <Earth_Explorer_Header>
    <Fixed_Header>
      <Mission>Sentinel-2B</Mission>
      <Ref_Frame>EARTH_FIXED</Ref_Frame>
      <Validity_Period>
        <Validity_Start>UTC=2024-06-01T00:00:00.000000</Validity_Start>
        <Validity_Stop>UTC=2024-06-01T23:59:59.000000</Validity_Stop>
      </Validity_Period>
    </Fixed_Header>
  </Earth_Explorer_Header>
  <Data_Block>
    <List_of_OSVs count="1">
      <OSV>
        <UTC>UTC=2024-06-01T06:30:00.000000</UTC>
        <Absolute_Orbit>99999</Absolute_Orbit>
        <X unit="m">7000000.000</X>
        <Y unit="m">0.000</Y>
        <Z unit="m">0.000</Z>
        <VX unit="m/s">0.000</VX>
        <VY unit="m/s">7500.000</VY>
        <VZ unit="m/s">0.000</VZ>
        <Quality>DEGRADED</Quality>
      </OSV>
    </List_of_OSVs>
  </Data_Block>
</Earth_Explorer_File>
)";

// =============================================================================
// parse_content tests
// =============================================================================

void test_eof_parse_empty_string() {
    // No validity period: refused, rather than defaulting the epoch to J2000.
    TEST_ASSERT_THROWS(EOFParser::parse_content(""), ballistic::Error);
}

void test_eof_parse_no_osvs() {
    std::string xml = R"(<?xml version="1.0"?>
<Earth_Explorer_File>
  <Earth_Explorer_Header>
    <Fixed_Header><Mission>Sentinel-1A</Mission></Fixed_Header>
  </Earth_Explorer_Header>
  <Data_Block>
    <List_of_OSVs count="0">
    </List_of_OSVs>
  </Data_Block>
</Earth_Explorer_File>)";

    TEST_ASSERT_THROWS(EOFParser::parse_content(xml), ballistic::Error);   // no validity
    const std::string with_validity = std::string(xml).replace(
        xml.find("</Fixed_Header>"), 0,
        "<Validity_Period><Validity_Start>UTC=2024-01-15T00:00:00.000000</Validity_Start>"
        "<Validity_Stop>UTC=2024-01-15T23:59:59.000000</Validity_Stop></Validity_Period>");
    TEST_ASSERT_MSG(!EOFParser::parse_content(with_validity).has_value(),
                    "XML with no OSVs should return nullopt");
}

void test_eof_parse_valid_2osv() {
    auto result = EOFParser::parse_content(VALID_2OSV_XML);
    TEST_ASSERT_MSG(result.has_value(), "Valid XML should parse successfully");
    TEST_ASSERT_MSG(result->mission == "Sentinel-1A", "Mission should be Sentinel-1A");
    TEST_ASSERT_MSG(result->state_vectors.size() == 2, "Should have 2 state vectors");
}

void test_eof_parse_mission() {
    auto result = EOFParser::parse_content(VALID_WITH_REF_FRAME_XML);
    TEST_ASSERT_MSG(result.has_value(), "Valid XML should parse");
    TEST_ASSERT_MSG(result->mission == "Sentinel-2B", "Mission should be Sentinel-2B");
}

void test_eof_parse_reference_frame() {
    auto result = EOFParser::parse_content(VALID_WITH_REF_FRAME_XML);
    TEST_ASSERT_MSG(result.has_value(), "Valid XML should parse");
    TEST_ASSERT_MSG(result->reference_frame == "EARTH_FIXED",
                "Reference frame should be EARTH_FIXED");
}

// =============================================================================
// Field verification tests
// =============================================================================

void test_eof_position_values() {
    auto result = EOFParser::parse_content(VALID_2OSV_XML);
    TEST_ASSERT_MSG(result.has_value(), "Should parse");

    const auto& sv = result->state_vectors[0];
    TEST_ASSERT_NEAR(sv.position.x(), 4500000.0, 0.01);
    TEST_ASSERT_NEAR(sv.position.y(), 3500000.0, 0.01);
    TEST_ASSERT_NEAR(sv.position.z(), 4000000.0, 0.01);
}

void test_eof_velocity_values() {
    auto result = EOFParser::parse_content(VALID_2OSV_XML);
    TEST_ASSERT_MSG(result.has_value(), "Should parse");

    const auto& sv = result->state_vectors[0];
    TEST_ASSERT_NEAR(sv.velocity.x(), -2500.0, 0.01);
    TEST_ASSERT_NEAR(sv.velocity.y(), 5000.0, 0.01);
    TEST_ASSERT_NEAR(sv.velocity.z(), 5500.0, 0.01);
}

void test_eof_epoch_parsing() {
    auto result = EOFParser::parse_content(VALID_2OSV_XML);
    TEST_ASSERT_MSG(result.has_value(), "Should parse");

    // First OSV: UTC=2024-01-15T12:00:00
    // Second OSV: UTC=2024-01-15T12:00:10
    double dt = result->state_vectors[1].epoch - result->state_vectors[0].epoch;
    TEST_ASSERT_NEAR(dt, 10.0, 0.01);
}

void test_eof_orbit_number() {
    auto result = EOFParser::parse_content(VALID_2OSV_XML);
    TEST_ASSERT_MSG(result.has_value(), "Should parse");
    TEST_ASSERT_MSG(result->state_vectors[0].orbit_number == 52001,
                "Orbit number should be 52001");
}

void test_eof_quality_flag() {
    auto result = EOFParser::parse_content(VALID_2OSV_XML);
    TEST_ASSERT_MSG(result.has_value(), "Should parse");
    TEST_ASSERT_MSG(result->state_vectors[0].quality == "NOMINAL",
                "Quality should be NOMINAL");

    auto result2 = EOFParser::parse_content(VALID_WITH_REF_FRAME_XML);
    TEST_ASSERT_MSG(result2.has_value(), "Should parse");
    TEST_ASSERT_MSG(result2->state_vectors[0].quality == "DEGRADED",
                "Quality should be DEGRADED");
}

// =============================================================================
// Malformed-data handling: a bad value must SKIP its OSV, never silently
// become a 0.0 (Earth-center) truth position or a J2000 epoch that would poison
// the RMSE (root-mean-square error) computed against the file.

// =============================================================================

void test_eof_malformed_position_skipped() {
    std::string xml = VALID_2OSV_XML;
    const std::string bad = "4499975.000";      // the 2nd OSV's X value
    auto pos = xml.find(bad);
    TEST_ASSERT_MSG(pos != std::string::npos, "fixture drifted");
    xml.replace(pos, bad.size(), "not_a_number");
    auto result = EOFParser::parse_content(xml);
    TEST_ASSERT_MSG(result.has_value(), "header valid — file still parses");
    TEST_ASSERT_MSG(result->state_vectors.size() == 1,
                "malformed OSV skipped, not zero-filled (would be 2 with 0.0)");
    TEST_ASSERT_MSG(result->skipped_records == 1, "the skip is counted");
    TEST_ASSERT_MSG(result->state_vectors[0].position.x() == 4500000.0,
                "the surviving OSV is the valid first one");
}

void test_eof_malformed_timestamp_skipped() {
    std::string xml = VALID_2OSV_XML;
    const std::string bad = "UTC=2024-01-15T12:00:10.000000";  // 2nd OSV UTC
    auto pos = xml.find(bad);
    TEST_ASSERT_MSG(pos != std::string::npos, "fixture drifted");
    xml.replace(pos, bad.size(), "UTC=garbage");
    auto result = EOFParser::parse_content(xml);
    TEST_ASSERT_MSG(result.has_value(), "file still parses");
    TEST_ASSERT_MSG(result->state_vectors.size() == 1,
                "OSV with malformed timestamp skipped, not j2000()-filled");
}

// =============================================================================
// Validity period tests
// =============================================================================

void test_eof_validity_period() {
    auto result = EOFParser::parse_content(VALID_2OSV_XML);
    TEST_ASSERT_MSG(result.has_value(), "Should parse");

    // Validity start: 2024-01-15T00:00:00
    // Validity stop:  2024-01-15T23:59:59
    double span = result->validity_stop - result->validity_start;
    // Should be ~86399 seconds (one day minus 1 second)
    TEST_ASSERT_NEAR(span, 86399.0, 1.0);
}

// =============================================================================
// find_nearest tests
// =============================================================================

void test_eof_find_nearest_empty() {
    EOFFile file;
    file.state_vectors.clear();

    Epoch target = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    auto result = file.find_nearest(target);
    TEST_ASSERT_MSG(!result.has_value(), "find_nearest on empty should return nullopt");
}

void test_eof_find_nearest_single() {
    auto parsed = EOFParser::parse_content(VALID_WITH_REF_FRAME_XML);
    TEST_ASSERT_MSG(parsed.has_value(), "Should parse");

    // Target very close to the single OSV epoch
    Epoch target = Epoch::from_utc(2024, 6, 1, 6, 30, 5);
    auto result = parsed->find_nearest(target);
    TEST_ASSERT_MSG(result.has_value(), "Should find nearest");
    TEST_ASSERT_NEAR(result->position.x(), 7000000.0, 0.01);
}

void test_eof_find_nearest_multiple() {
    auto parsed = EOFParser::parse_content(VALID_2OSV_XML);
    TEST_ASSERT_MSG(parsed.has_value(), "Should parse");

    // Target at 12:00:08 — closer to second OSV at 12:00:10 than first at 12:00:00
    Epoch target = Epoch::from_utc(2024, 1, 15, 12, 0, 8);
    auto result = parsed->find_nearest(target);
    TEST_ASSERT_MSG(result.has_value(), "Should find nearest");
    TEST_ASSERT_NEAR(result->position.x(), 4499975.0, 0.01);
}

// =============================================================================
// get_range tests
// =============================================================================

void test_eof_get_range() {
    auto parsed = EOFParser::parse_content(VALID_2OSV_XML);
    TEST_ASSERT_MSG(parsed.has_value(), "Should parse");

    // Range that includes both OSVs
    Epoch start = Epoch::from_utc(2024, 1, 15, 11, 59, 0);
    Epoch end = Epoch::from_utc(2024, 1, 15, 12, 1, 0);
    auto range = parsed->get_range(start, end);
    TEST_ASSERT_MSG(range.size() == 2, "Should get both OSVs in range");

    // Narrow range that excludes second OSV
    Epoch end2 = Epoch::from_utc(2024, 1, 15, 12, 0, 5);
    auto range2 = parsed->get_range(start, end2);
    TEST_ASSERT_MSG(range2.size() == 1, "Should get only first OSV");
}

// =============================================================================
// interpolate_position tests
// =============================================================================

void test_eof_interpolate_insufficient_data() {
    auto parsed = EOFParser::parse_content(VALID_2OSV_XML);
    TEST_ASSERT_MSG(parsed.has_value(), "Should parse");

    // Default order is 9, but we only have 2 OSVs
    Epoch target = Epoch::from_utc(2024, 1, 15, 12, 0, 5);
    auto result = parsed->interpolate_position(target);
    TEST_ASSERT_MSG(!result.has_value(),
                "Interpolation with insufficient data should return nullopt");
}

void test_eof_interpolate_low_order() {
    auto parsed = EOFParser::parse_content(VALID_2OSV_XML);
    TEST_ASSERT_MSG(parsed.has_value(), "Should parse");

    // With order=2, should succeed with 2 OSVs
    Epoch target = Epoch::from_utc(2024, 1, 15, 12, 0, 5);
    auto result = parsed->interpolate_position(target, 2);
    TEST_ASSERT_MSG(result.has_value(),
                "Order-2 interpolation with 2 OSVs should succeed");

    // Result should be between the two positions
    TEST_ASSERT_MSG(result->x() < 4500000.0 && result->x() > 4499975.0,
                "Interpolated X should be between the two OSV positions");

    // Outside the state vectors: no value (no extrapolation).
    TEST_ASSERT_MSG(!parsed->interpolate_position(Epoch::from_utc(2024, 1, 15, 12, 0, 30), 2),
                    "no extrapolation past the last OSV");
    TEST_ASSERT_MSG(!parsed->interpolate_position(Epoch::from_utc(2024, 1, 15, 11, 59, 0), 2),
                    "no extrapolation before the first OSV");
}

// =============================================================================
// Tag extraction correctness — X vs VX
// =============================================================================

void test_eof_x_not_confused_with_vx() {
    // Verify that parsing X doesn't accidentally pick up VX values
    auto result = EOFParser::parse_content(VALID_2OSV_XML);
    TEST_ASSERT_MSG(result.has_value(), "Should parse");

    const auto& sv = result->state_vectors[0];
    // X = 4500000, VX = -2500 — these must not be confused
    TEST_ASSERT_MSG(std::abs(sv.position.x() - 4500000.0) < 1.0,
                "X should be 4500000, not VX value");
    TEST_ASSERT_MSG(std::abs(sv.velocity.x() - (-2500.0)) < 1.0,
                "VX should be -2500, not X value");
}

// =============================================================================
// Zip extraction
// =============================================================================

void test_eof_failed_extraction_leaves_no_nul_file() {
    // "2>nul" is a Windows device name; on POSIX a shell creates a regular
    // file called "nul" in the working directory. A failed extraction must
    // leave the working directory untouched.
    namespace fs = std::filesystem;
    const fs::path scratch = fs::temp_directory_path() / "ballistic_eof_nul_test";
    fs::remove_all(scratch);
    fs::create_directories(scratch);
    const fs::path original_cwd = fs::current_path();
    fs::current_path(scratch);
    const auto result = extract_eof_zip((scratch / "does_not_exist.zip").string());
    // The working directory must still be empty. (Not fs::exists(scratch /
    // "nul"): on Windows "nul" is the device and that call throws.)
    const bool untouched = fs::is_empty(scratch);
    fs::current_path(original_cwd);
    fs::remove_all(scratch);
    TEST_ASSERT_MSG(!result.has_value(), "a nonexistent zip cannot be extracted");
    TEST_ASSERT_MSG(untouched, "extraction left a file (e.g. 'nul') in the working directory");
}

// =============================================================================
// Main
// =============================================================================

int main() {
    TestRunner parse_tests("EOF parse_content");
    parse_tests.add_test("Empty string", test_eof_parse_empty_string);
    parse_tests.add_test("No OSVs", test_eof_parse_no_osvs);
    parse_tests.add_test("Valid 2-OSV XML", test_eof_parse_valid_2osv);
    parse_tests.add_test("Mission parsing", test_eof_parse_mission);
    parse_tests.add_test("Reference frame", test_eof_parse_reference_frame);

    TestRunner field_tests("EOF field verification");
    field_tests.add_test("Position values", test_eof_position_values);
    field_tests.add_test("Velocity values", test_eof_velocity_values);
    field_tests.add_test("Epoch parsing", test_eof_epoch_parsing);
    field_tests.add_test("Orbit number", test_eof_orbit_number);
    field_tests.add_test("Quality flag", test_eof_quality_flag);
    field_tests.add_test("Malformed position skipped", test_eof_malformed_position_skipped);
    field_tests.add_test("Malformed timestamp skipped", test_eof_malformed_timestamp_skipped);
    field_tests.add_test("Validity period", test_eof_validity_period);

    TestRunner query_tests("EOF queries");
    query_tests.add_test("find_nearest empty", test_eof_find_nearest_empty);
    query_tests.add_test("find_nearest single", test_eof_find_nearest_single);
    query_tests.add_test("find_nearest multiple", test_eof_find_nearest_multiple);
    query_tests.add_test("get_range", test_eof_get_range);

    TestRunner interp_tests("EOF interpolation");
    interp_tests.add_test("Insufficient data", test_eof_interpolate_insufficient_data);
    interp_tests.add_test("Low order interpolation", test_eof_interpolate_low_order);

    TestRunner tag_tests("EOF tag extraction");
    tag_tests.add_test("X not confused with VX", test_eof_x_not_confused_with_vx);

    TestRunner zip_tests("EOF zip extraction");
    zip_tests.add_test("Failed extraction leaves no 'nul' file",
                       test_eof_failed_extraction_leaves_no_nul_file);

    parse_tests.run();
    field_tests.run();
    query_tests.run();
    interp_tests.run();
    tag_tests.run();
    zip_tests.run();

    return print_final_summary();
}
