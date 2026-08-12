#!/bin/sh

set -eu

package="ness-agent"
command_name="ness"
uv_version="0.12.3"
uv_installer_url="https://releases.astral.sh/github/uv/releases/download/$uv_version/uv-installer.sh"
uv_installer_sha256="a7e3924ea1cd06bf1518c577d635c624ae2e2db030e0fc8ff8cf426224384e17"
pypi_index_url="https://pypi.org/simple"
install_log=""
uv_installer=""

if [ -t 1 ] && [ "${TERM:-}" != "dumb" ]; then
  bold=$(printf '\033[1m')
  dim=$(printf '\033[2m')
  green=$(printf '\033[32m')
  red=$(printf '\033[31m')
  reset=$(printf '\033[0m')
else
  bold=""
  dim=""
  green=""
  red=""
  reset=""
fi

cleanup() {
  if [ -n "$install_log" ]; then
    rm -f "$install_log"
  fi
  if [ -n "$uv_installer" ]; then
    rm -f "$uv_installer"
  fi
}

interrupted() {
  cleanup
  trap - 0
  exit 130
}

fail() {
  printf '\n%s%sInstallation failed.%s\n' "$bold" "$red" "$reset" >&2
  printf '%s\n' "$1" >&2
  exit 1
}

step() {
  printf '%s==>%s %s\n' "$bold" "$reset" "$1"
}

ready() {
  printf '    %sok%s  %s\n' "$green" "$reset" "$1"
}

warning() {
  printf '    %swarning%s  %s\n' "$red" "$reset" "$1" >&2
}

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    checksum_output=$(sha256sum "$1") || return 1
  elif command -v shasum >/dev/null 2>&1; then
    checksum_output=$(shasum -a 256 "$1") || return 1
  else
    return 1
  fi
  printf '%s\n' "${checksum_output%% *}"
}

trap cleanup 0
trap interrupted HUP INT TERM

if [ -z "${HOME:-}" ]; then
  fail "HOME is not set. Set HOME to your user directory, then run this installer again."
fi
home_dir=$HOME

if [ -z "${PATH:-}" ]; then
  fail "PATH is not set. Set PATH, then run this installer again."
fi
current_path=$PATH

uv_install_dir=${UV_INSTALL_DIR:-"$home_dir/.local/bin"}
case "$uv_install_dir" in
  /*) ;;
  *) fail "UV_INSTALL_DIR must be an absolute path: $uv_install_dir" ;;
esac

case "$(uname -s 2>/dev/null || printf unknown)" in
  Darwin|Linux) ;;
  *) fail "This installer currently supports macOS and Linux." ;;
esac

if ! command -v curl >/dev/null 2>&1; then
  fail "curl is required. Install curl, then run this command again."
fi

install_log=$(mktemp "${TMPDIR:-/tmp}/ness-agent-install.XXXXXX") ||
  fail "Could not create a temporary install log."

printf '\n%sNESS AGENT%s\n' "$bold" "$reset"
printf '%sown the loop%s\n\n' "$dim" "$reset"

if command -v uv >/dev/null 2>&1; then
  uv_bin=$(command -v uv)
  ready "uv found"
elif [ -x "$uv_install_dir/uv" ]; then
  uv_bin="$uv_install_dir/uv"
  ready "uv found at $uv_bin"
else
  step "Installing uv $uv_version"
  uv_installer=$(mktemp "${TMPDIR:-/tmp}/ness-agent-uv.XXXXXX") ||
    fail "Could not create a temporary uv installer."

  if ! curl --proto '=https' --proto-redir '=https' --tlsv1.2 -LsSf \
    "$uv_installer_url" -o "$uv_installer"; then
    fail "Could not download uv from $uv_installer_url"
  fi

  if ! actual_sha256=$(file_sha256 "$uv_installer"); then
    fail "Could not verify the uv installer; sha256sum or shasum is required."
  fi
  if [ "$actual_sha256" != "$uv_installer_sha256" ]; then
    fail "uv installer checksum mismatch. Expected $uv_installer_sha256, got $actual_sha256."
  fi

  # Use a deterministic location so a custom UV_INSTALL_DIR is honored and the
  # executable can be located without parsing installer output.
  unset UV_UNMANAGED_INSTALL
  if ! UV_INSTALL_DIR="$uv_install_dir" UV_NO_MODIFY_PATH=1 \
    sh "$uv_installer" >"$install_log" 2>&1; then
    printf '\n%s%suv installation failed.%s\n\n' "$bold" "$red" "$reset" >&2
    cat "$install_log" >&2
    exit 1
  fi

  rm -f "$uv_installer"
  uv_installer=""

  uv_bin="$uv_install_dir/uv"
  if [ ! -x "$uv_bin" ]; then
    fail "uv was installed, but its executable was not found at $uv_bin"
  fi

  ready "uv installed"
fi

step "Installing or updating Ness Agent"
: >"$install_log"

# Pin the tool environment to Python 3.12. uv downloads it automatically when
# the machine does not already have a compatible interpreter. --upgrade makes
# rerunning this installer refresh and select the newest compatible release.
# Ignore ambient uv indexes/configuration so the public package is resolved
# only from PyPI over HTTPS.
unset UV_CONFIG_FILE UV_DEFAULT_INDEX UV_EXTRA_INDEX_URL UV_FIND_LINKS
unset UV_INDEX UV_INDEX_URL UV_INSECURE_HOST UV_NO_INDEX
if ! "$uv_bin" tool install --python 3.12 --upgrade --no-config \
  --default-index "$pypi_index_url" "$package" >"$install_log" 2>&1; then
  printf '\n%s%sNess Agent installation failed.%s\n\n' "$bold" "$red" "$reset" >&2
  cat "$install_log" >&2
  exit 1
fi

bin_dir=$("$uv_bin" tool dir --bin 2>/dev/null || true)
if [ -z "$bin_dir" ] || [ ! -x "$bin_dir/$command_name" ]; then
  fail "The install completed, but the ness executable could not be located."
fi

case ":$current_path:" in
  *":$bin_dir:"*) path_ready=1 ;;
  *) path_ready=0 ;;
esac

path_update_failed=0
if [ "$path_ready" -eq 0 ]; then
  : >"$install_log"
  if ! "$uv_bin" tool update-shell --no-config >"$install_log" 2>&1; then
    path_update_failed=1
    warning "Could not update your shell PATH automatically."
    if [ -s "$install_log" ]; then
      cat "$install_log" >&2
    fi
    printf '    Ness was installed at: %s\n' "$bin_dir/$command_name" >&2
  fi
fi

ready "Ness Agent is up to date"
printf '\n%sReady.%s ' "$bold" "$reset"
if [ "$path_ready" -eq 1 ]; then
  printf 'Run it with:\n\n  %sness%s\n\n' "$bold" "$reset"
elif [ "$path_update_failed" -eq 1 ]; then
  printf 'Run it now with:\n\n  %s%s%s\n\n' "$bold" "$bin_dir/$command_name" "$reset"
  printf 'Or add this directory to PATH:\n\n  %sexport PATH="%s:$PATH"%s\n\n' \
    "$dim" "$bin_dir" "$reset"
else
  printf 'Restart your terminal, then run:\n\n  %sness%s\n\n' "$bold" "$reset"
  printf '%sInstalled at %s%s\n\n' "$dim" "$bin_dir/$command_name" "$reset"
fi
