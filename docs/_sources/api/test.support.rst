test.support
============

.. automodule:: test.support

   
   .. rubric:: Functions

   .. autosummary::
   
      adjust_int_max_str_digits
      anticipate_failure
      args_from_interpreter_flags
      bigaddrspacetest
      bigmemtest
      busy_retry
      calcobjsize
      calcvobjsize
      captured_output
      captured_stderr
      captured_stdin
      captured_stdout
      check__all__
      check_bolt_optimized
      check_cflags_pgo
      check_disallow_instantiation
      check_free_after_iterating
      check_impl_detail
      check_sanitizer
      check_sizeof
      check_syntax_error
      clear_ignored_deprecations
      collision_stats
      copy_python_src_ignore
      cpython_only
      darwin_malloc_err_warning
      detect_api_mismatch
      disable_faulthandler
      disable_gc
      findfile
      flush_std_streams
      gc_collect
      get_attribute
      get_original_stdout
      get_pagesize
      get_recursion_available
      get_recursion_depth
      has_no_debug_ranges
      ignore_deprecations_from
      impl_detail
      infinite_recursion
      is_resource_enabled
      iter_builtin_types
      iter_slot_wrappers
      late_deletion
      load_package_tests
      maybe_get_event_loop_policy
      missing_compiler_executable
      no_tracing
      open_urlresource
      optim_args_from_interpreter_flags
      patch
      patch_list
      print_warning
      python_is_optimized
      reap_children
      record_original_stdout
      refcount_test
      requires
      requires_bz2
      requires_debug_ranges
      requires_fork
      requires_freebsd_version
      requires_gzip
      requires_legacy_unicode_capi
      requires_limited_api
      requires_linux_version
      requires_lzma
      requires_mac_ver
      requires_resource
      requires_specialization
      requires_subprocess
      requires_venv_with_pip
      requires_working_socket
      requires_zlib
      run_code
      run_in_subinterp
      run_in_subinterp_with_config
      run_with_locale
      run_with_locales
      run_with_tz
      set_memlimit
      set_recursion_limit
      set_sanitizer_env_var
      setswitchinterval
      setup_venv_with_pip_setuptools_wheel
      skip_if_broken_multiprocessing_synchronize
      skip_if_buggy_ucrt_strfptime
      skip_if_buildbot
      skip_if_pgo_task
      skip_if_sanitizer
      sleeping_retry
      sortdict
      suppress_msvcrt_asserts
      swap_attr
      swap_item
      system_must_validate_cert
      wait_process
      with_pymalloc
   
   .. rubric:: Classes

   .. autosummary::
   
      BrokenIter
      Matcher
      PythonSymlink
      SaveSignals
      SuppressCrashReport
      catch_unraisable_exception
   
   .. rubric:: Exceptions

   .. autosummary::
   
      Error
      ResourceDenied
      TestDidNotRun
      TestFailed
      TestFailedWithDetails
   
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:

   ast_helper
   asynchat
   asyncore
   bytecode_helper
   hashlib_helper
   hypothesis_helper
   i18n_helper
   import_helper
   interpreters
   logging_helper
   os_helper
   pty_helper
   script_helper
   smtpd
   socket_helper
   testcase
   threading_helper
   venv
   warnings_helper
