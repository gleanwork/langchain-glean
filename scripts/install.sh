TASK_VERSION="v3.36.0"
BIN_PATH=""
SYS=$(uname -s)

set -e

function install_gotask {
	if which task >/dev/null 2>&1; then
		echo "task found in \$PATH. Skipping go-task installation."
		return
	fi

	if [[ "$SYS" = "Darwin" && -z "${GITHUB_PATH}" ]]; then
		# local non-GHA mac
		cat <<HERE
This MP uses go-task <https://taskfile.dev>.
On MacOS install go-task with:

  arch -arm64 brew install go-task/tap/go-task

If you are still on an intel mac, please request a new computer, or run \`brew
install\` without \`arch\`.

For further details consult the documentation at <https://taskfile.dev/installation/>
HERE
		exit 1
	fi

	ARCH=$(uname -m)
	DIST_ARCH=""
	if [ "$ARCH" = "x86_64" ]; then
		DIST_ARCH="amd64"
	else
		cat <<HERE
Unable to determine go-task download URL.  Using:
SYS=$SYS
ARCH=$ARCH
DIST_ARCH=$DIST_ARCH
HERE
		exit 1
	fi

	SYS_LOWER=$(echo "$SYS" | tr 'L' 'l')
	TASK_URL="https://github.com/go-task/task/releases/download/${TASK_VERSION}/task_${SYS_LOWER}_${DIST_ARCH}.tar.gz"
	TASK_DOWNLOAD_BIN="$HOME/task"
	echo "fetch ${TASK_URL}"

  curl -L "${TASK_URL}" -o "${HOME}/task.tar.gz"
  (cd "$HOME" && tar xfz "${HOME}/task.tar.gz")

  if [[ ! -f "$TASK_DOWNLOAD_BIN" ]]; then
    echo "Expecting task at '$TASK_DOWNLOAD_BIN' but no file is there.  Check"
    exit 1
  fi

	if [ -n "${GITHUB_PATH}" ]; then
		# Must use $HOME/bin for post-merge (see comment below)
		BIN_PATH="$HOME/bin"

		# Running in GitHub Action, we can update path with $GITHUB_PATH

		# This updates $PATH in subsequent GitHub actions steps.
		echo "$BIN_PATH" >>"$GITHUB_PATH"
	elif [[ "$SYS" = "Linux" ]]; then
    echo "linux; presumably rdev"
    BIN_PATH="/usr/local/bin"
	else
		# Fallback to printing env/debugging info
		#
		cat <<HERE
I don't know where I'm running.  I'm going to print \`env\` and \$PATH for debugging.
HERE

		env
		echo "$PATH"

		exit 1
	fi

	mkdir -p "$BIN_PATH"
  mv "$TASK_DOWNLOAD_BIN" "$BIN_PATH/task" 2> /dev/null ||
    sudo mv "$TASK_DOWNLOAD_BIN" "$BIN_PATH/task"
}

function run_task_setup {
	if which task >/dev/null 2>&1; then
		# local mint setup probably
		task setup
	else
		# CI mint setup
		#
		# We're in the same step so we can't rely on $BIN_PATH having been added to
		# $PATH on e.g. darwin GHA runners.
		"$BIN_PATH/task" -o prefixed "setup"
	fi
}

install_gotask
run_task_setup
