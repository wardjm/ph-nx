defmodule PhNx.CLITest do
  use ExUnit.Case
  import ExUnit.CaptureIO

  defp tmp_file(contents) do
    path =
      Path.join(System.tmp_dir!(), "ph_nx_cli_test_#{System.unique_integer([:positive])}.txt")

    File.write!(path, contents)
    path
  end

  describe "main/1 - argument parsing" do
    test "prints usage to stdout with --help" do
      output = capture_io(fn -> PhNx.CLI.main(["--help"]) end)
      assert output =~ "Usage:"
      assert output =~ "--max-dim"
    end

    test "errors on unknown option" do
      output =
        capture_io(:stderr, fn ->
          catch_exit(PhNx.CLI.main(["--unknown"]))
        end)

      assert output =~ "Unknown option"
    end

    test "errors when no file is given" do
      output =
        capture_io(:stderr, fn ->
          catch_exit(PhNx.CLI.main([]))
        end)

      assert output =~ "no input file"
    end

    test "errors when multiple files are given" do
      output =
        capture_io(:stderr, fn ->
          catch_exit(PhNx.CLI.main(["a.txt", "b.txt"]))
        end)

      assert output =~ "too many arguments"
    end
  end

  describe "main/1 - file handling" do
    test "errors with a friendly message when file does not exist" do
      output =
        capture_io(:stderr, fn ->
          catch_exit(PhNx.CLI.main(["/nonexistent/path/points.txt"]))
        end)

      assert output =~ "Error reading"
    end

    test "errors with a friendly message on empty file" do
      path = tmp_file("# just a comment\n\n")

      output =
        capture_io(:stderr, fn ->
          catch_exit(PhNx.CLI.main([path]))
        end)

      File.rm(path)
      assert output =~ "no data points"
    end

    test "errors on invalid coordinate" do
      path = tmp_file("0.0,0.0\n1.0,bad\n")

      output =
        capture_io(:stderr, fn ->
          catch_exit(PhNx.CLI.main([path]))
        end)

      File.rm(path)
      assert output =~ "invalid number on line 2"
    end

    test "processes a valid point cloud file" do
      path =
        tmp_file("""
        # square
        0.0,0.0
        1.0,0.0
        1.0,1.0
        0.0,1.0
        """)

      output =
        capture_io(fn ->
          PhNx.CLI.main([path])
        end)

      File.rm(path)
      assert output =~ "Computing persistent homology for 4 points in 2D"
      assert output =~ "Betti numbers"
    end

    test "accepts tab-separated coordinates" do
      path = tmp_file("0.0\t0.0\n1.0\t0.0\n1.0\t1.0\n0.0\t1.0\n")

      output =
        capture_io(fn ->
          PhNx.CLI.main([path])
        end)

      File.rm(path)
      assert output =~ "4 points in 2D"
    end

    test "respects --max-dim option" do
      path =
        tmp_file("0.0,0.0\n1.0,0.0\n1.0,1.0\n0.0,1.0\n")

      output =
        capture_io(fn ->
          PhNx.CLI.main([path, "--max-dim", "1"])
        end)

      File.rm(path)
      assert output =~ "Computing"
    end
  end
end
