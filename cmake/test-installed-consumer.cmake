set(_prefix "${ANNDATA_BUILD_DIR}/installed-test/prefix")
set(_consumer "${ANNDATA_BUILD_DIR}/installed-test/consumer")
function(checked)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE _result OUTPUT_VARIABLE _stdout ERROR_VARIABLE _stderr)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Installed consumer command failed (${_result}): ${ARGV}\n${_stdout}\n${_stderr}")
  endif()
endfunction()
checked("${CMAKE_COMMAND}" --install "${ANNDATA_BUILD_DIR}" --prefix "${_prefix}" --config "${ANNDATA_CONFIG}")
checked("${CMAKE_COMMAND}" -S "${ANNDATA_SOURCE_DIR}/tests/consumer" -B "${_consumer}" -G Ninja
  "-DCMAKE_CXX_COMPILER=${ANNDATA_COMPILER}" "-DCMAKE_BUILD_TYPE=${ANNDATA_CONFIG}"
  "-DCMAKE_PREFIX_PATH=${_prefix}")
checked("${CMAKE_COMMAND}" --build "${_consumer}")
checked("${_consumer}/consumer")
